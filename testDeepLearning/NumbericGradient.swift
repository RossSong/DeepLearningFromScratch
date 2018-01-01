//
//  NumbericGradient.swift
//  testDeepLearning
//
//  Created by RossSong on 2018. 1. 1..
//  Copyright © 2018년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

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
