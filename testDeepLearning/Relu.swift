//
//  Relu.swift
//  testDeepLearning
//
//  Created by RossSong on 2018. 1. 1..
//  Copyright © 2018년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

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

