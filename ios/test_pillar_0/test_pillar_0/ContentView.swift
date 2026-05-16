//
//  ContentView.swift
//  test_pillar_0
//
//  Created by Sai Teja Ranuva on 5/16/26.
//

import CoreML
import SwiftUI

struct ValidationMetrics {
    let finite: Bool
    let cosineSimilarity: Double
    let maxAbsDiff: Float
    let meanAbsDiff: Double
    let expectedNorm: Double
    let actualNorm: Double
    let loadSeconds: TimeInterval
    let predictionSeconds: TimeInterval
    let outputCount: Int

    var passed: Bool {
        finite && cosineSimilarity >= 0.999 && maxAbsDiff <= 0.01
    }
}

@MainActor
final class Pillar0ValidationModel: ObservableObject {
    @Published var status = "Ready"
    @Published var metrics: ValidationMetrics?
    @Published var errorMessage: String?
    @Published var isRunning = false

    func run() {
        guard !isRunning else { return }
        isRunning = true
        status = "Loading bundled Core ML model"
        metrics = nil
        errorMessage = nil

        Task.detached(priority: .userInitiated) {
            do {
                let metrics = try Self.validate()
                print(
                    "PILLAR0_VALIDATION passed=\(metrics.passed) " +
                    "cosine=\(metrics.cosineSimilarity) " +
                    "max_abs_diff=\(metrics.maxAbsDiff) " +
                    "prediction_sec=\(metrics.predictionSeconds)"
                )
                await MainActor.run {
                    self.metrics = metrics
                    self.status = metrics.passed ? "Passed" : "Failed thresholds"
                    self.isRunning = false
                }
            } catch {
                print("PILLAR0_VALIDATION_ERROR \(String(describing: error))")
                await MainActor.run {
                    self.errorMessage = String(describing: error)
                    self.status = "Error"
                    self.isRunning = false
                }
            }
        }
    }

    nonisolated private static func validate() throws -> ValidationMetrics {
        guard let resourceURL = Bundle.main.resourceURL else {
            throw ValidationError("Bundle resource URL is missing")
        }

        let resources = resourceURL.appendingPathComponent("Pillar0Resources", isDirectory: true)
        let modelURL = resources.appendingPathComponent("Pillar0HeadCTVision_int8.mlmodelc", isDirectory: true)
        let inputURL = resources.appendingPathComponent("simulator_sample_windowed_headct_f16.raw")
        let expectedURL = resources.appendingPathComponent("simulator_expected_torchscript_f32.raw")

        let config = MLModelConfiguration()
#if targetEnvironment(simulator)
        config.computeUnits = .cpuOnly
#else
        config.computeUnits = .cpuAndNeuralEngine
#endif

        let loadStarted = Date()
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        let loadSeconds = Date().timeIntervalSince(loadStarted)

        let inputArray = try MLMultiArray(shape: [1, 11, 128, 256, 256], dataType: .float16)
        try copyRawBytes(inputURL, into: inputArray)
        let expected = try readFloat32Raw(expectedURL, expectedCount: 1152)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "windowed_headct": MLFeatureValue(multiArray: inputArray)
        ])

        let predictionStarted = Date()
        let output = try model.prediction(from: provider)
        let predictionSeconds = Date().timeIntervalSince(predictionStarted)

        guard let outputArray = output.featureValue(for: "var_4269")?.multiArrayValue else {
            throw ValidationError("Missing Core ML output var_4269")
        }

        return try compare(
            expected: expected,
            actual: floatArray(from: outputArray),
            loadSeconds: loadSeconds,
            predictionSeconds: predictionSeconds
        )
    }

    nonisolated private static func copyRawBytes(_ url: URL, into multiArray: MLMultiArray) throws {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        let byteCount: Int
        switch multiArray.dataType {
        case .float16:
            byteCount = multiArray.count * MemoryLayout<UInt16>.stride
        case .float32:
            byteCount = multiArray.count * MemoryLayout<Float>.stride
        default:
            throw ValidationError("Unsupported input MLMultiArray dtype \(multiArray.dataType.rawValue)")
        }
        guard data.count == byteCount else {
            throw ValidationError("Expected \(byteCount) input bytes, got \(data.count)")
        }
        let destination = multiArray.dataPointer.bindMemory(to: UInt8.self, capacity: data.count)
        _ = data.copyBytes(to: UnsafeMutableBufferPointer(start: destination, count: data.count))
    }

    nonisolated private static func readFloat32Raw(_ url: URL, expectedCount: Int) throws -> [Float] {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        let expectedBytes = expectedCount * MemoryLayout<Float>.stride
        guard data.count == expectedBytes else {
            throw ValidationError("Expected \(expectedBytes) output bytes, got \(data.count)")
        }
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }

    nonisolated private static func floatArray(from multiArray: MLMultiArray) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(multiArray.count)
        for index in 0..<multiArray.count {
            values.append(multiArray[index].floatValue)
        }
        return values
    }

    nonisolated private static func compare(
        expected: [Float],
        actual: [Float],
        loadSeconds: TimeInterval,
        predictionSeconds: TimeInterval
    ) throws -> ValidationMetrics {
        guard expected.count == actual.count else {
            throw ValidationError("Expected output count \(expected.count), got \(actual.count)")
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

        return ValidationMetrics(
            finite: finite,
            cosineSimilarity: dot / (sqrt(expectedNorm) * sqrt(actualNorm)),
            maxAbsDiff: maxAbsDiff,
            meanAbsDiff: sumAbsDiff / Double(actual.count),
            expectedNorm: sqrt(expectedNorm),
            actualNorm: sqrt(actualNorm),
            loadSeconds: loadSeconds,
            predictionSeconds: predictionSeconds,
            outputCount: actual.count
        )
    }
}

struct ValidationError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

struct ContentView: View {
    @StateObject private var validation = Pillar0ValidationModel()

    var body: some View {
        NavigationStack {
            List {
                Section("Status") {
                    HStack {
                        Text(validation.status)
                        Spacer()
                        if validation.isRunning {
                            ProgressView()
                        } else if let metrics = validation.metrics {
                            Image(systemName: metrics.passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(metrics.passed ? .green : .red)
                        }
                    }

                    Button(validation.isRunning ? "Running..." : "Run Pillar-0 Validation") {
                        validation.run()
                    }
                    .disabled(validation.isRunning)
                }

                if let metrics = validation.metrics {
                    Section("Metrics") {
                        metric("Cosine", String(format: "%.9f", metrics.cosineSimilarity))
                        metric("Max abs diff", String(format: "%.9f", metrics.maxAbsDiff))
                        metric("Mean abs diff", String(format: "%.9f", metrics.meanAbsDiff))
                        metric("Finite", metrics.finite ? "true" : "false")
                        metric("Output count", "\(metrics.outputCount)")
                        metric("Load", String(format: "%.2fs", metrics.loadSeconds))
                        metric("Prediction", String(format: "%.2fs", metrics.predictionSeconds))
                        metric("Expected norm", String(format: "%.6f", metrics.expectedNorm))
                        metric("Actual norm", String(format: "%.6f", metrics.actualNorm))
                    }
                }

                if let errorMessage = validation.errorMessage {
                    Section("Error") {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Pillar-0 HeadCT")
            .task {
                validation.run()
            }
        }
    }

    private func metric(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .fontDesign(.monospaced)
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    ContentView()
}
