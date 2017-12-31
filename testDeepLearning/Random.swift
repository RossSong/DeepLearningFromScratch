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
    // stored properties
    var s : Double = 0.0
    var v2 : Double = 0.0
    var cachedNumberExists = false
    
    // (read-only) computed properties
    var gaussRand : Double  {
        var u1, u2, v1, x : Double
        if !cachedNumberExists {
            repeat {
                u1 = Double(arc4random()) / Double(UINT32_MAX)
                u2 = Double(arc4random()) / Double(UINT32_MAX)
                v1 = 2 * u1 - 1;
                v2 = 2 * u2 - 1;
                s = v1 * v1 + v2 * v2;
            } while (s >= 1 || s == 0)
            x = v1 * sqrt(-2 * log(s) / s);
        }
        else {
            x = v2 * sqrt(-2 * log(s) / s);
        }
        cachedNumberExists = !cachedNumberExists
        return x
    }
    
    class func randn(inputSize: Int, outputSize: Int) -> Tensor<Double> {
//        let generator = MyRandomGenerator()
//
//        generator.s = mean
//        generator.v2 = std
        let n = Distributions.Normal(m: 0, v: 1)
        let tensor = Tensor<Double>(dimensions: [inputSize, outputSize])
        for i in 0..<inputSize {
            for j in 0..<outputSize {
//                tensor[i,j] = generator.gaussRand
                tensor[i, j] = n.Random()
            }
        }
        
        return tensor
    }
}
