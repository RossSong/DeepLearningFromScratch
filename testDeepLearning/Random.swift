//
//  Random.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 12. 19..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

//Gaussian Distribution Random

class MyRandomGenerator {
    class func randn(inputSize: Int, outputSize: Int) -> Tensor<Double> {
        let n = Distributions.Normal(m: 0, v: 1)
        let tensor = Tensor<Double>(dimensions: [inputSize, outputSize])
        for i in 0..<inputSize {
            for j in 0..<outputSize {
                tensor[i, j] = n.Random()
            }
        }
        
        return tensor
    }
}
