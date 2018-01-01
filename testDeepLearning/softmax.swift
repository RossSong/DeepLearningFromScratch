//
//  softmax.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 11. 23..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func doSoftMax(elements: ValueArray<Double>) -> ValueArray<Double> {
    let c = max(elements)
    let exp_a = exp(elements - c)
    let sum_exp_a = sum(exp_a)
    let y = exp_a / sum_exp_a
    return y
}

func softmax(_ a:Tensor<Double>) -> Tensor<Double> {
    let b = a.copy()
    for i in 0..<a.dimensions[0] {
        let startIndex = (i * a.dimensions[1])
        let endIndex = ((i + 1) * a.dimensions[1]) - 1
        let array = ValueArray<Double>(a.elements[startIndex...endIndex])
        let ret = doSoftMax(elements: array)
        
        for j in 0..<ret.count {
            b.elements[startIndex + j] = ret[j]
        }
    }
    
    return b
}
