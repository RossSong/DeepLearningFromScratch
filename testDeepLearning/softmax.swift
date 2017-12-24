//
//  softmax.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 11. 23..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func softmax(_ a:Tensor<Double>) -> Tensor<Double> {
    let c = max(a.elements)
    let exp_a = exp(a.elements - c)
    let sum_exp_a = sum(exp_a)
    let y = exp_a / sum_exp_a
    
    return Tensor<Double>(Matrix(rows: a.dimensions[0], columns: a.dimensions[1], elements: y))
}
