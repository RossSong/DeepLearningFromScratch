//
//  Activation.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation

func sigmoid(_ x: Double) -> Double {
    return 1.0 / ( 1.0 + exp(-1 * x))
}

func sigmoid(_ x: [Double]) -> [Double] {
    var result = Array<Double>(repeating: 0.0, count: x.count)
    for i in 0..<x.count {
        result[i] = sigmoid(x[i])
    }
    return result
}

func relu(_ x: Double) -> Double {
    return 0.0 < x ? x : 0.0
}

func relu(_ x: [Double]) -> [Double] {
    var result = Array<Double>(repeating: 0.0, count: x.count)
    for i in 0..<x.count {
        result[i] = relu(x[i])
    }
    return result
}
