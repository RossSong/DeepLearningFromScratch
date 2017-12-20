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

func argmax(x: Tensor<Double>) -> Tensor<Double> {
    var max = 0.0
    var index = 0
    let y = x.copy()
    
    //가장 큰 것만 1 나머지는 0
    for i in 0..<x.elements.count {
        if x.elements[i] > max {
            max = x.elements[i]
            index = i
        }
        
        y.elements[i] = 0.0
    }
    
    if y.elements.count >= index {
        debugPrint("argmax error!!! out of index")
    }
    
    y.elements[index] = 1.0
    return y
}

func zeroLike(x: Tensor<Double>) -> ValueArray<Double> {
    let ret = ValueArray<Double>(count: x.elements.count, repeatedValue: 0.0)
    return ret
}

func boolEqualTensorSum(x: Tensor<Double>, y: Tensor<Double>) -> Double {
    var total: Double = 0
    for i in 0..<x.elements.count {
        if x[i] == y[i] {
            total = total + 1
        }
    }
    
    return total
}

func numericalGradient(f: ((_ x: Tensor<Double>) -> Double), x: Tensor<Double>) -> Tensor<Double>{
    let h: Double = 0.0001
    let grad = zeroLike(x: x)
    
    for idx in 0..<x.elements.count {
        let tmpVal = x.elements[idx]
        x.elements[idx] = tmpVal + h
        let fxh1 = f(x)
        
        x.elements[idx] = tmpVal - h
        let fxh2 = f(x)
        
        let tmp = (fxh1 - fxh2) / (2 * h)
        grad[idx] = tmp
        x.elements[idx] = tmpVal
    }
    
    return Tensor<Double>(grad.toRowMatrix())
}

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
    
    func backward(dout: Tensor<Double>) -> Tensor<Double>{
        guard let x = self.x else { debugPrint("error"); return Tensor<Double>(Matrix<Double>([[0.0]])) }
        
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
        
        return Tensor<Double>(dx)
    }
}

func crossEntropyError(y: Tensor<Double>, t: Tensor<Double>) -> Double {
    let batch_size: Double = Double(y.dimensions[0])
    return -1 * Upsurge.sum(t.elements * Upsurge.log(y.elements)) / batch_size
}

class SoftMaxWithLoss {
    var loss: Double = None
    var y: Tensor<Double>?
    var t: Tensor<Double>?

    func forward(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        self.t = t
        let y = softmax(x)
        self.y = y
        loss = crossEntropyError(y: y, t: t)
        return loss
    }

    func backward(dout: Double = 1) -> Tensor<Double> {
        guard let y = y,  let t = t else { return Tensor<Double>(dimensions: [1]) }
        let batch_size = Double(t.dimensions[0])
        let dx = (y.elements - t.elements) / batch_size
        return Tensor<Double>(dx.toRowMatrix())
    }
}

class Network2 {
    var params: Dictionary<String, Tensor<Double>>!
    var layers:  [String: Any]!
    var lastLayer: SoftMaxWithLoss!

    init(inputSize: Int, hiddeSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
        params = Dictionary<String, Tensor<Double>>()
        
        let randW1 = MyRandomGenerator.randn(inputSize: inputSize, outputSize: hiddeSize)
        let weightedRandW1 = weightInitStd * randW1.elements
        params["W1"] = Tensor<Double>(weightedRandW1.toRowMatrix())
        params["b1"] = Tensor<Double>(ValueArray<Double>(count: hiddeSize, repeatedValue: 0.0).toRowMatrix())
        
        let randW2 = MyRandomGenerator.randn(inputSize: hiddeSize, outputSize: outputSize)
        let weightedRandW2 = weightInitStd * randW2.elements
        params["W2"] = Tensor<Double>(weightedRandW2.toRowMatrix())
        params["b2"] = Tensor<Double>(ValueArray<Double>(count: outputSize, repeatedValue: 0.0).toRowMatrix())

        self.layers = [String: Any]()
        self.layers["Affine1"] = Affine(W: params["W1"]!, b: params["b1"]!)
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(W: params["W2"]!, b: params["b2"]!)

        self.lastLayer = SoftMaxWithLoss()
    }
    
    func predict(x: Tensor<Double>) -> Tensor<Double> {
        var x: Tensor<Double>?
        
        let layerAffine1 = self.layers["Affine1"] as! Affine
        x = layerAffine1.forward(x: x!)
        let layerRelu1 = self.layers["Relu1"] as! Relu
        x = layerRelu1.forward(x: x!)
        let layerAffine2 = self.layers["Affine2"] as! Affine
        x = layerAffine2.forward(x: x!)
        
        return x!
    }
    
    func loss(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        let y = predict(x: x)
        return self.lastLayer.forward(x:y, t:t)
    }
    
    func accuracy(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        var y = self.predict(x: x)
        var t = t
        let newY = argmax(x: y)
        
        if 1 != t.rank {
            t = argmax(x: t)
        }

//        let accuracy = Upsurge.sum((y == t) / x.dimensions[0]
        let accuracy = boolEqualTensorSum(x: newY, y: t) / Double(x.dimensions[0])
        return accuracy
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
        
        let tmp = ValueArray<Double>(count: t.count, repeatedValue: 1.0)
        var dout: Tensor<Double> = Tensor<Double>(tmp.toRowMatrix())
        
        let layerAffine1 = self.layers["Affine2"] as! Affine
        dout = layerAffine1.backward(dout: dout)
        let layerRelu1 = self.layers["Relu1"] as! Relu
        dout = layerRelu1.backward(dout: dout)
        let layerAffine2 = self.layers["Affine1"] as! Affine
        dout = layerAffine2.backward(dout: dout)
        
        var grads = [String: Any]()
        grads["W1"] = (self.layers["Affine1"] as! Affine).dW
        grads["b1"] = (self.layers["Affine1"] as! Affine).db
        grads["W2"] = (self.layers["Affine2"] as! Affine).dW
        grads["b2"] = (self.layers["Affine2"] as! Affine).db
        
        return grads
    }
}


