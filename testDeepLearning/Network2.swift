//
//  neuron.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

let None = -1

class Relu {
    func forward(x: Tensor<Double>) -> Tensor<Double> {
        let out = x.copy()
        
        for i in 0..<out.count {
            if out[i] <= 0 {
                out[i] = 0
            }
        }
        
        return out
    }
    
    func backward(dout: Tensor<Double>) -> Tensor<Double> {
        for i in 0..<dout.count {
            if dout[i] <= 0 {
                dout[i] = 0
            }
        }

        return dout
    }
}

class Affine {
    var W: Tensor<Double>
    var b: Tensor<Double>
    var x: Tensor<Double>?
    var dW: Tensor<Double>?
    var db: Tensor<Double>?
    
    init(W: Tensor<Double>, b: Tensor<Double>) {
        self.W = W
        self.b = b
    }
    
    func forward(x: Tensor<Double>) -> Tensor<Double> {
        self.x = x
        
        let tx = x.asMatrix(0...x.dimensions[0], 0...x.dimensions[1])
        let tW = W.asMatrix(0...W.dimensions[0], 0...W.dimensions[1])
        let tb = b.asMatrix(0...b.dimensions[0], 0...b.dimensions[1])
        
        let out = tx * tW + tb
        return Tensor<Double>(out)
    }
    
    func backward(dout: Tensor<Double>) -> Upsurge.Matrix<Double>{
        guard let x = self.x else { debugPrint("error"); return Matrix<Double>([[0.0]]) }
        
        let tDout = dout.asMatrix(0...dout.dimensions[0], 0...dout.dimensions[1])
        let transposeW = self.W.asMatrix(0...self.W.dimensions[1], 0...self.W.dimensions[0])
        let transposeX = x.asMatrix(0...x.dimensions[1], 0...x.dimensions[0])
        
        let dx = tDout * transposeW
        self.dW = Tensor(transposeX * tDout)
        
        let tDb = Tensor<Double>(dimensions: [tDout.dimensions[0], 1])
        
        for i in 0..<tDout.dimensions[0] {
            var total: Double = 0
            for j in 0..<tDout.dimensions[1] {
                total = total + tDout[i,j]
            }
            tDb[i] = total
        }
        
        self.db = tDb
        
        return dx
    }
}

func crossEntropyError(y: Tensor<Double>, t: Tensor<Double>) -> Double {
    let batch_size: Double = Double(y.dimensions[0])
    return -1 * Upsurge.sum(t.elements * Upsurge.log(y.elements)) / batch_size
}

//class SoftMaxWithLoss {
//    var loss: Double = None
//    var y: Double = None
//    var t: Double = None
//
//    func foward(x, t) -> Double {
//        t = t
//        y = softmax(x)
//        loss = crossEntropyError(y, t)
//        return loss
//    }
//
//    func backward(dout = 1) {
//        batch_size = t.shape[0]
//        dx = (y - t) / batch_size
//        return dx
//    }
//}
//
//class Network2 {
//    var params: Dictionary<String, Tensor<Double>>!
//    var layers:  [Int: [String: Any]]?
//    var lastLayer: Any?
//
//    init(inputSize: Int, hiddeSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
//        params = Dictionary<String, Tensor<Double>>()
//        params["W1"] = weightInitStd * MyRandomGenerator.randn(inputSize: inputSize, outputSize: hiddeSize)
//        params["b1"] = Tensor<Double>(rows: 1, columns: hiddeSize, repeatedValue: 0.0)
//        params["W2"] = weightInitStd * MyRandomGenerator.randn(inputSize: hiddeSize, outputSize: outputSize)
//        params["b2"] = Tensor<Double>(rows: 1, columns: outputSize, repeatedValue: 0.0)
//
//        self.layers = Dictionary<Int, Dictionary<String, Any>>()
//        self.layers![0]!["Affine1"] = Affine(W: params!["W1"], b: params!["b1"])
//        self.layers![1]!["Relu1"] = Relu()
//        self.layers![2]!["Affine2"] = Affine(W: params!["W2"], b: params!["b2"])
//
//        self.lastLayer = SoftMaxWithLoss()
//    }
//}

