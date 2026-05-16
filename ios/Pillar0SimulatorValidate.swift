import CoreML
import Foundation

struct ValidationError: Error, CustomStringConvertible {
    let description: String
}

func fail(_ message: String) -> Never {
    fputs("error: \(message)\n", stderr)
    exit(1)
}

func computeUnits(_ value: String) -> MLComputeUnits {
    switch value {
    case "cpuOnly":
        return .cpuOnly
    case "cpuAndGPU":
        return .cpuAndGPU
    case "cpuAndNeuralEngine":
        return .cpuAndNeuralEngine
    case "all":
        return .all
    default:
        fail("unknown computeUnits \(value)")
    }
}

func readFloat32Raw(_ url: URL, expectedCount: Int) throws -> [Float] {
    let data = try Data(contentsOf: url, options: [.mappedIfSafe])
    let expectedBytes = expectedCount * MemoryLayout<Float>.stride
    guard data.count == expectedBytes else {
        throw ValidationError(description: "expected \(expectedBytes) bytes in \(url.path), got \(data.count)")
    }
    return data.withUnsafeBytes { rawBuffer in
        Array(rawBuffer.bindMemory(to: Float.self))
    }
}

func copyRawBytes(_ url: URL, into multiArray: MLMultiArray) throws {
    let data = try Data(contentsOf: url, options: [.mappedIfSafe])
    let byteCount: Int
    switch multiArray.dataType {
    case .float16:
        byteCount = multiArray.count * MemoryLayout<UInt16>.stride
    case .float32:
        byteCount = multiArray.count * MemoryLayout<Float>.stride
    default:
        throw ValidationError(description: "unsupported input MLMultiArray dtype \(multiArray.dataType.rawValue)")
    }
    guard data.count == byteCount else {
        throw ValidationError(description: "expected \(byteCount) bytes in \(url.path), got \(data.count)")
    }
    let destination = multiArray.dataPointer.bindMemory(to: UInt8.self, capacity: data.count)
    _ = data.copyBytes(to: UnsafeMutableBufferPointer(start: destination, count: data.count))
}

func floatArray(from multiArray: MLMultiArray) -> [Float] {
    var values = [Float]()
    values.reserveCapacity(multiArray.count)
    for index in 0..<multiArray.count {
        values.append(multiArray[index].floatValue)
    }
    return values
}

func metrics(expected: [Float], actual: [Float]) throws -> [String: Any] {
    guard expected.count == actual.count else {
        throw ValidationError(description: "expected output count \(expected.count), got \(actual.count)")
    }
    var dot: Double = 0
    var expectedNorm: Double = 0
    var actualNorm: Double = 0
    var maxAbsDiff: Float = 0
    var sumAbsDiff: Double = 0
    var finite = true

    for index in expected.indices {
        let lhs = expected[index]
        let rhs = actual[index]
        finite = finite && lhs.isFinite && rhs.isFinite
        let diff = abs(lhs - rhs)
        maxAbsDiff = max(maxAbsDiff, diff)
        sumAbsDiff += Double(diff)
        dot += Double(lhs) * Double(rhs)
        expectedNorm += Double(lhs) * Double(lhs)
        actualNorm += Double(rhs) * Double(rhs)
    }

    let cosine = dot / (sqrt(expectedNorm) * sqrt(actualNorm))
    return [
        "finite": finite,
        "output_count": actual.count,
        "cosine_similarity": cosine,
        "max_abs_diff": maxAbsDiff,
        "mean_abs_diff": sumAbsDiff / Double(actual.count),
        "expected_norm": sqrt(expectedNorm),
        "actual_norm": sqrt(actualNorm),
    ]
}

func jsonPrint(_ object: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

let args = CommandLine.arguments
guard args.count >= 4 else {
    fail("usage: Pillar0SimulatorValidate <model.mlmodelc> <input_f16.raw> <expected_f32.raw> [cpuOnly|cpuAndNeuralEngine|cpuAndGPU|all]")
}

let modelURL = URL(fileURLWithPath: args[1])
let inputURL = URL(fileURLWithPath: args[2])
let expectedURL = URL(fileURLWithPath: args[3])
let units = args.count >= 5 ? args[4] : "cpuAndNeuralEngine"

let inputShape: [NSNumber] = [1, 11, 128, 256, 256]
let outputCount = 1152
let inputName = "windowed_headct"
let outputName = "var_4269"

do {
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits(units)

    let loadStarted = Date()
    let model = try MLModel(contentsOf: modelURL, configuration: config)
    let loadSeconds = Date().timeIntervalSince(loadStarted)

    let inputArray = try MLMultiArray(shape: inputShape, dataType: .float16)
    try copyRawBytes(inputURL, into: inputArray)
    let expected = try readFloat32Raw(expectedURL, expectedCount: outputCount)

    let provider = try MLDictionaryFeatureProvider(dictionary: [
        inputName: MLFeatureValue(multiArray: inputArray)
    ])

    let predictionStarted = Date()
    let output = try model.prediction(from: provider)
    let predictionSeconds = Date().timeIntervalSince(predictionStarted)

    guard let outputArray = output.featureValue(for: outputName)?.multiArrayValue else {
        throw ValidationError(description: "missing output \(outputName)")
    }

    var result = try metrics(expected: expected, actual: floatArray(from: outputArray))
    result["model"] = modelURL.path
    result["input"] = inputURL.path
    result["expected"] = expectedURL.path
    result["compute_units"] = units
    result["load_sec"] = loadSeconds
    result["prediction_sec"] = predictionSeconds
    result["input_shape"] = inputShape.map { $0.intValue }
    result["output_shape"] = outputArray.shape.map { $0.intValue }
    jsonPrint(result)
} catch {
    fail(String(describing: error))
}
