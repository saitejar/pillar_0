import CoreML
import Foundation

final class Pillar0HeadCTRunner {
    static let inputName = "windowed_headct"
    static let outputName = "var_4269"
    static let inputShape = [1, 11, 128, 256, 256]

    private let model: MLModel

    init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }

    func predict(windowedHeadCT: MLMultiArray) throws -> MLMultiArray {
        precondition(
            windowedHeadCT.shape.map { $0.intValue } == Self.inputShape,
            "Expected Pillar HeadCT tensor shape \(Self.inputShape)"
        )

        let input = try MLDictionaryFeatureProvider(dictionary: [
            Self.inputName: MLFeatureValue(multiArray: windowedHeadCT)
        ])

        let output = try model.prediction(from: input)
        guard let embedding = output.featureValue(for: Self.outputName)?.multiArrayValue else {
            throw NSError(
                domain: "Pillar0HeadCTRunner",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Missing output \(Self.outputName)"]
            )
        }
        return embedding
    }
}
