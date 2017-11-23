//
//  softmax.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 11. 23..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func softmax(_ a:ValueArray<Double>) -> ValueArray<Double> {
    let c = max(a)
    let exp_a = exp(a - c)
    let sum_exp_a = sum(exp_a)
    let y = exp_a / sum_exp_a
    
    return y
}
