import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import Forge

/**
  The neural network from the paper "MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications" https://arxiv.org/abs/1704.04861v1
*/
class MobileNet: NeuralNetwork {
  typealias Prediction = (labelIndex: Int, probability: Float)

  let classes: Int
  let model: Model

  /**
    Creates a new MobileNet object.
    
    - Parameters:
      - widthMultiplier: Shrinks the number of channels. This is a value in the
        range (0, 1]. Default is 1, which starts the network with 32 channels.
        (This hyperparameter is called "alpha" in the paper.)
      - resolutionMultiplier: Shrink the spatial dimensions of the input image.
        This is a value between (0, 1]. Default is 1, which resizes to 224x224
        pixels. (The paper calls this hyperparameter "rho".)
      - shallow: Whether to exclude the group of 5 conv layers in the middle.
      - classes: The number of classes in the softmax.
  */
  public init(device: MTLDevice,
              widthMultiplier: Float = 1,
              resolutionMultiplier: Float = 1,
              shallow: Bool = false,
              classes: Int = 1000,
              inflightBuffers: Int) {

    self.classes = classes

    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let sigmoid = MPSCNNNeuronSigmoid(device: device)

    let channels = Int(32 * widthMultiplier)
    let resolution = Int(256 * resolutionMultiplier)

    let input = Input()

    var x = input
        --> Resize(width: 256, height: 512)
        --> Custom(Preprocessing(device: device), channels: 3)
        --> Convolution(kernel: (3, 3), channels: channels, stride: (2, 2), activation: relu, name: "conv1")
        --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_1")
        --> PointwiseConvolution(channels: channels*2, activation: relu, name: "conv_pw_1")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv_dw_2")
        --> PointwiseConvolution(channels: channels*4, activation: relu, name: "conv_pw_2")
        --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_3")
        --> PointwiseConvolution(channels: channels*4, activation: relu, name: "conv_pw_3")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv_dw_4")
        --> PointwiseConvolution(channels: channels*8, activation: relu, name: "conv_pw_4")
        --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_5")
        --> PointwiseConvolution(channels: channels*8, activation: relu, name: "conv_pw_5")
        --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv_dw_6")
        --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv_pw_6")

    if !shallow {
      x = x --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_7")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv_pw_7")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_8")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv_pw_8")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_9")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv_pw_9")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_10")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv_pw_10")
            --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_11")
            --> PointwiseConvolution(channels: channels*16, activation: relu, name: "conv_pw_11")
    }

    x = x --> DepthwiseConvolution(kernel: (3, 3), stride: (2, 2), activation: relu, name: "conv_dw_12")
          --> PointwiseConvolution(channels: channels*32, activation: relu, name: "conv_pw_12")
          --> DepthwiseConvolution(kernel: (3, 3), activation: relu, name: "conv_dw_13")
          --> PointwiseConvolution(channels: channels*32, activation: relu, name: "conv_pw_13")
  
    var parts = x --> ConvolutionTranspose(kernel: (3, 3), channels: 14, stride: (2, 2), activation: sigmoid, name: "conv2d_transpose_1")
    var locref = x --> ConvolutionTranspose(kernel: (3, 3), channels: 28, stride: (2, 2), activation: nil, name: "conv2d_transpose_2")

    var outputs: [Tensor] = [parts, locref]
    
    
    
    model = Model(input: input, output: outputs)

    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderBundle(name: name,
                                                 count: count,
                                                 suffix: type == .weights ? "_W" : "_b",
                                                 ext: "bin")
    }

    if success {
      print(model.summary())
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture inputTexture: MTLTexture, inflightIndex: Int) {
    model.encode(commandBuffer: commandBuffer, texture: inputTexture, inflightIndex: inflightIndex)
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    let parts = model.outputImage(index: 0, inflightIndex: inflightIndex).toFloatArray()
    let locref = model.outputImage(index: 1, inflightIndex: inflightIndex).toFloatArray()

    var result = NeuralNetworkResult<Prediction>()
    return result
  }
}
