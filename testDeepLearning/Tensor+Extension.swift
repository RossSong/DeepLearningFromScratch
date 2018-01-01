//
//  Tensor+Extension.swift
//  testDeepLearning
//
//  Created by RossSong on 2018. 1. 1..
//  Copyright © 2018년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

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

func *(lhs: ValueArray<Double>, rhs: Tensor<Double>) -> Tensor<Double> {
    let left = lhs.toRowMatrix()
    let right = Matrix<Double>(rows: rhs.dimensions[0], columns: rhs.dimensions[1], elements: rhs.elements)
    let ret = left * right
    return Tensor<Double>(ret)
}

func *(lhs: Tensor<Double>, rhs: ValueArray<Double>) -> Tensor<Double> {
    let left = Matrix<Double>(rows: lhs.dimensions[0], columns: lhs.dimensions[1], elements: lhs.elements)
    let right = rhs.toColumnMatrix()
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

extension Tensor {
    var T: Tensor<Element> {
        get {
            let m = Tensor<Element>(Matrix<Element>(rows: self.dimensions[1], columns: self.dimensions[0]))
            for i in 0..<self.dimensions[0] {
                for j in 0..<self.dimensions[1] {
                    m[j, i] = self[i, j]
                }
            }
            
            return m
        }
    }
}

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

func toMatrix(_ x: Tensor<Double>) -> Matrix<Double> {
    return Matrix<Double>(rows: x.dimensions[0], columns: x.dimensions[1], elements: x.elements)
}
