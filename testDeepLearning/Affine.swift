//
//  Affine.swift
//  testDeepLearning
//
//  Created by RossSong on 2018. 1. 1..
//  Copyright © 2018년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

class Affine {
    var W: Tensor<Double>
    var b: Tensor<Double>
    var x: Tensor<Double>?
    var dW: Tensor<Double>?
    var db: Double = 0.0
    
    init(W: Tensor<Double>, b: Tensor<Double>) {
        self.W = W
        self.b = b
    }
    
    func forward(x: Tensor<Double>) -> Tensor<Double> {
        self.x = x
        let out = x * W + b
        return out
    }
    
    func backward(dout: Tensor<Double>) -> Tensor<Double> {
        let dx = dout * self.W.T
        self.dW = self.x!.T * dout
        self.db = sumA(dout.elements)
        return dx
    }
}
