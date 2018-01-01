//
//  TwoLayerNet.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 12. 31..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge


func sigmoid(_ x: Tensor<Double>) -> Tensor<Double> {
    let ret = x.copy()
    for i in 0..<ret.dimensions[0] {
        for j in 0..<ret.dimensions[1] {
            ret[i, j] = sigmoid(ret[i, j])
        }
    }
    
    return ret
}

func sigmoid_grad(x: Tensor<Double>) -> Tensor<Double> {
    let tmp = sigmoid(x)
    for i in 0..<tmp.elements.count {
        tmp.elements[i] = (1.0 - tmp.elements[i]) * tmp.elements[i]
    }
    
    return tmp
}

class TwoLayerNet {
    var params: Dictionary<String, Tensor<Double>>
    
    init(inputSize: Int, hiddenSize: Int, outputSize: Int, weightInitStd: Double = 0.01)  {
        self.params = Dictionary<String, Tensor<Double>>()
        let randW1 = MyRandomGenerator.randn(inputSize: inputSize, outputSize: hiddenSize)
        let weightedRandW1 = weightInitStd * randW1.elements
        params["W1"] = Tensor<Double>(Matrix<Double>(rows: inputSize, columns: hiddenSize, elements: weightedRandW1))
        params["b1"] = Tensor<Double>(ValueArray<Double>(count: hiddenSize, repeatedValue: 0.0).toRowMatrix())
        
        let randW2 = MyRandomGenerator.randn(inputSize: hiddenSize, outputSize: outputSize)
        let weightedRandW2 = weightInitStd * randW2.elements
        params["W2"] = Tensor<Double>(Matrix<Double>(rows: hiddenSize, columns: outputSize, elements: weightedRandW2))
        params["b2"] = Tensor<Double>(ValueArray<Double>(count: outputSize, repeatedValue: 0.0).toRowMatrix())
    }

    func predict(x: Tensor<Double>) -> Tensor<Double> {
        let W1 = self.params["W1"]
        let W2 = self.params["W2"]
        let b1 = self.params["b1"]
        let b2 = self.params["b2"]

        let a1 = (x * W1!) + b1!
        let z1 = sigmoid(a1)
        let a2 = z1 * W2! + b2!
        let y = softmax(a2)
        return y
    }

    func loss(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        let y = self.predict(x: x)
        return crossEntropyError(y: y, t: t)
    }

    func accuracy(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        let y = self.predict(x: x)
        var t = t
        let newY = argmax(x: y)
        
        if 1 != t.rank {
            t = argmax(x: t)
        }
        
        let acc = boolEqualTensorSum(x: newY, y: t)
        return acc
    }

    func numericalGradientA(x: Tensor<Double>, t: Tensor<Double>) -> [String: Any] {
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
        let W1 = self.params["W1"]
        let W2 = self.params["W2"]
        let b1 = self.params["b1"]
        let b2  = self.params["b2"]
        var grads = [String: Any]()
        let batchNum: Double = Double(x.dimensions[0])

        // forward
        let a1 = (x * W1!) + b1!
        let z1 = sigmoid(a1)
        let a2 = z1 * W2! + b2!
        let y = softmax(a2)

        // backward
        let dy = (y - t) / batchNum
        grads["W2"] = transpose(z1) * dy
        grads["b2"] = sumA(dy.elements)//Upsurge.sum(dy.elements)//sum(dy)

        let da1 = dy * transpose(W2!)
        let dz1 = sigmoid_grad(x: a1).elements * da1.elements
        
        grads["W1"] = transpose(x) * Tensor<Double>(dz1.toMatrix(rows: a1.dimensions[0], columns: a1.dimensions[1]))
        grads["b1"] = sumA(dz1)//Upsurge.sum(dz1)//sum(Tensor<Double>(Matrix<Double>(rows: da1.dimensions[0], columns: da1.dimensions[1], elements: dz1)))

        return grads
    }
}

func sumA(_ x: ValueArray<Double>) -> Double {
    var total: Double = 0.0
    for i in 0..<x.count {
        total = total + x[i]
    }
    
    return total
}

func *(lhs: Tensor<Double>, rhs: Tensor<Double>) -> Tensor<Double> {
    let left = Matrix<Double>(rows: lhs.dimensions[0], columns: lhs.dimensions[1], elements: lhs.elements)
    let right = Matrix<Double>(rows: rhs.dimensions[0], columns: rhs.dimensions[1], elements: rhs.elements)
    let ret = left * right
    return Tensor<Double>(ret)
}

func +(lhs: Tensor<Double>, rhs: Tensor<Double>) -> Tensor<Double> {
    let out = lhs.copy()
    
    for i in 0..<out.dimensions[0] {
        for j in 0..<out.dimensions[1] {
            out[i, j] = out[i, j] + rhs.elements[j]
        }
    }
    
    return out
}

func -(lhs: Tensor<Double>, rhs: Tensor<Double>) -> Tensor<Double> {
    let out = lhs.copy()
    
    for i in 0..<out.dimensions[0] {
        for j in 0..<out.dimensions[1] {
            out[i, j] = out[i, j] - rhs.elements[j]
        }
    }
    
    return out
}

func *(lhs: Double, rhs: Tensor<Double>) -> Tensor<Double> {
    let right = Matrix<Double>(rows: rhs.dimensions[0], columns: rhs.dimensions[1], elements: rhs.elements)
    let ret = right.elements * lhs
    return Tensor<Double>(Matrix<Double>(rows: rhs.dimensions[0], columns: rhs.dimensions[1], elements: ret))
}

func /(lhs: Tensor<Double>, rhs: Double) -> Tensor<Double> {
    let left = Matrix<Double>(rows: lhs.dimensions[0], columns: lhs.dimensions[1], elements: lhs.elements)
    let ret = left.elements / rhs
    return Tensor<Double>(Matrix<Double>(rows: lhs.dimensions[0], columns: lhs.dimensions[1], elements: ret))
}

func -(lhs: Double, rhs: Tensor<Double>) -> Tensor<Double> {
    let right = Matrix<Double>(rows: rhs.dimensions[0], columns: rhs.dimensions[1], elements: rhs.elements)
    let ret = -1 * right.elements + lhs
    return Tensor<Double>(Matrix<Double>(rows: rhs.dimensions[0], columns: rhs.dimensions[1], elements: ret))
}

