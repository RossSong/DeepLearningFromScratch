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
    let array = ValueArray<Double>(capacity: x.dimensions[0])
    
    for i in 0..<x.dimensions[0] {
        var max = 0.0
        var index = 0
        for j in 0..<x.dimensions[1] {
            if x[i, j] > max {
                max = x[i, j]
                index = j
            }
        }
        
        array.append(Double(index))
    }
    
    return Tensor<Double>(Matrix<Double>(rows: 1, columns: x.dimensions[0], elements: array))
}

func zeroLike(x: Tensor<Double>) -> ValueArray<Double> {
    let ret = ValueArray<Double>(count: x.elements.count, repeatedValue: 0.0)
    return ret
}

func boolEqualTensorSum(x: Tensor<Double>, y: Tensor<Double>) -> Double {
    var total: Double = 0
    
    for i in 0..<x.elements.count {
        if x.elements[i] == y.elements[i] {
            total = total + 1
        }
    }
    
    return total / Double(x.elements.count)
}

func numericalGradient(f: ((_ x: Tensor<Double>) -> Double), x: Tensor<Double>) -> Tensor<Double>{
    let h: Double = 1e-4//0.0001
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

func gradientDescent(f: ((_ x: Tensor<Double>) -> Double), initX: Tensor<Double>, lr: Double = 0.01, stepNum: Int  = 100) -> Tensor<Double> {
    let x = initX
    
    for _ in 0..<stepNum {
        let grad = numericalGradient(f: f, x: x)
        x.elements = x.elements - lr * grad.elements
    }
    
    return x
}

class Relu {
    var dx: Tensor<Double>?
    
    func mask(out: Tensor<Double>, i: Int, j: Int) {
        if out.elements[i * out.dimensions[1] + j] < 0 {
            out.elements[i * out.dimensions[1] + j] = 0
        }
    }
    
    func forward(x: Tensor<Double>) -> Tensor<Double> {
        let out = x.copy()
        
        for i in 0..<out.dimensions[0] {
            for j in 0..<out.dimensions[1] {
                mask(out: out, i: i, j: j)
            }
        }
        
        return out
    }
    
    func backward(dout: Tensor<Double>) -> Tensor<Double> {
        for i in 0..<dout.dimensions[0] {
            for j in 0..<dout.dimensions[1] {
                mask(out: dout, i: i, j: j)
            }
        }

        dx = dout
        return dout
    }
}

func toMatrix(_ x: Tensor<Double>) -> Matrix<Double> {
    return Matrix<Double>(rows: x.dimensions[0], columns: x.dimensions[1], elements: x.elements)
}

func transpose(_ x: Tensor<Double>) -> Tensor<Double> {
    
    let m = Tensor<Double>(Matrix(rows: x.dimensions[1], columns: x.dimensions[0]))
    for i in 0..<x.dimensions[0] {
        for j in 0..<x.dimensions[1] {
            m[j, i] = x[i, j]
        }
    }
    
    return m
}

class Affine {
    var W: Tensor<Double>
    var b: Tensor<Double>
    var x: Tensor<Double>?
    var dW: Tensor<Double>?
    var db: Double = 0.0
    
    init(W: Tensor<Double>, b: Tensor<Double>) {
        self.W = W
        self.b = b
    }
    
    func forward(x: Tensor<Double>) -> Tensor<Double> {
        self.x = x
        
        let tx = Matrix<Double>(rows:x.dimensions[0], columns:x.dimensions[1], elements: x.elements)
        let tW = Matrix<Double>(rows:W.dimensions[0], columns:W.dimensions[1], elements: W.elements)
        let tb = b.elements
        
        let out = Tensor<Double>(tx * tW)
        
        //(tx * tw) + tb -> out + tb
        for i in 0..<out.dimensions[0] {
            for j in 0..<out.dimensions[1] {
                out[i, j] = out[i, j] + tb[j]
            }
        }
        
        return out
    }
    
    func backward(dout: Tensor<Double>) -> Tensor<Double> {
        let transposeW = toMatrix(transpose(W))
        let transposeX = toMatrix(transpose(x!))
        let mDout = toMatrix(dout)
        
        let dx = mDout * transposeW
        let tmp = transposeX * mDout
        self.dW = Tensor(tmp)
        
        self.db = sumA(dout.elements)
        
        return Tensor<Double>(dx)
    }
}

func crossEntropyError(y: Tensor<Double>, t: Tensor<Double>) -> Double {
    let batch_size: Double = Double(y.dimensions[0])
    if y.elements.count != t.elements.count || 0 == batch_size {
        debugPrint("Error - crossEntropyError")
    }
    
    let val = Upsurge.sum(t.elements * Upsurge.log(y.elements))
    if 0 == val {
        return 0
    }
    
    return -1 * val / batch_size
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
        let batch_size = Double(t.elements.count)
        let dx = (y.elements - t.elements) / batch_size
        return Tensor<Double>(Matrix(rows: y.dimensions[0], columns: y.dimensions[1], elements:dx))
    }
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

//        let accuracy = Upsurge.sum((y == t) / x.dimensions[0]
        let acc = boolEqualTensorSum(x: newY, y: t) / Double(x.dimensions[0])
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


