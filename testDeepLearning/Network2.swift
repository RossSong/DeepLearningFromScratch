//
//  neuron.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

let None: Double = -1

func boolEqualTensorSum(x: Tensor<Double>, y: Tensor<Double>) -> Double {
    var total: Double = 0
    
    for i in 0..<x.elements.count {
        if x.elements[i] == y.elements[i] {
            total = total + 1
        }
    }
    
    return total / Double(x.elements.count)
}

class Network2 {
    var params: Dictionary<String, Tensor<Double>>!
    var layers:  [String: Any]!
    var lastLayer: SoftMaxWithLoss!
    var countOfClass = 10

    init(inputSize: Int, hiddeSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
        params = Dictionary<String, Tensor<Double>>()
        
        let randW1 = MyRandomGenerator.randn(inputSize: inputSize, outputSize: hiddeSize)
        let weightedRandW1 = weightInitStd * randW1.elements
        params["W1"] = Tensor<Double>(Matrix<Double>(rows: inputSize, columns: hiddeSize, elements: weightedRandW1))
        params["b1"] = Tensor<Double>(ValueArray<Double>(count: hiddeSize, repeatedValue: 0.0).toRowMatrix())
        
        let randW2 = MyRandomGenerator.randn(inputSize: hiddeSize, outputSize: outputSize)
        let weightedRandW2 = weightInitStd * randW2.elements
        params["W2"] = Tensor<Double>(Matrix<Double>(rows: hiddeSize, columns: outputSize, elements: weightedRandW2))
        params["b2"] = Tensor<Double>(ValueArray<Double>(count: outputSize, repeatedValue: 0.0).toRowMatrix())

        self.layers = [String: Any]()
        self.layers["Affine1"] = Affine(W: params["W1"]!, b: params["b1"]!)
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(W: params["W2"]!, b: params["b2"]!)

        self.lastLayer = SoftMaxWithLoss()
        
        countOfClass = outputSize
    }
    
    func predict(_ xVal: Tensor<Double>) -> Tensor<Double> {
        var x: Tensor<Double> = xVal
        
        let layerAffine1 = self.layers["Affine1"] as! Affine
        x = layerAffine1.forward(x: x)
        let layerRelu1 = self.layers["Relu1"] as! Relu
        x = layerRelu1.forward(x: x)
        let layerAffine2 = self.layers["Affine2"] as! Affine
        x = layerAffine2.forward(x: x)
        
        return x
    }
    
    func loss(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        let y = predict(x)
        return self.lastLayer.forward(x:y, t:t)
    }
    
    func accuracy(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        let y = self.predict(x)
        var t = t
        let newY = argmax(x: y)
        
        if 1 != t.rank {
            t = argmax(x: t)
        }
        
        let acc = boolEqualTensorSum(x: newY, y: t)
        return acc
    }

    func numerical_gradient(x: Tensor<Double>, t: Tensor<Double>) -> [String: Any] {
        func lossW(W: Tensor<Double>) -> Double {
            return loss(x: x, t: t)
        }
        
        var grads = [String: Any]()
        grads["W1"] = numericalGradient(f: lossW, x: self.params["W1"]!)
        grads["b1"] = numericalGradient(f: lossW, x: self.params["b1"]!)
        grads["W2"] = numericalGradient(f: lossW, x: self.params["W2"]!)
        grads["b2"] = numericalGradient(f: lossW, x: self.params["b2"]!)
        
        return grads
    }
    
    func gradient(x: Tensor<Double>, t: Tensor<Double>) -> [String: Any] {
        _ = self.loss(x: x, t: t)
        
        var dout = self.lastLayer.backward(dout: 1)
        let layerAffine2 = self.layers["Affine2"] as! Affine
        dout = layerAffine2.backward(dout: dout)
        let layerRelu1 = self.layers["Relu1"] as! Relu
        dout = layerRelu1.backward(dout: dout)
        let layerAffine1 = self.layers["Affine1"] as! Affine
        dout = layerAffine1.backward(dout: dout)
        
        var grads = [String: Any]()
        grads["W1"] = (self.layers["Affine1"] as! Affine).dW
        grads["b1"] = (self.layers["Affine1"] as! Affine).db
        grads["W2"] = (self.layers["Affine2"] as! Affine).dW
        grads["b2"] = (self.layers["Affine2"] as! Affine).db
        
        return grads
    }
}


