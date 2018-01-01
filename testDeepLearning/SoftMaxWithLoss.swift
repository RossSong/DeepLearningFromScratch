//
//  SoftMaxWithLoss.swift
//  testDeepLearning
//
//  Created by RossSong on 2018. 1. 1..
//  Copyright © 2018년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

class SoftMaxWithLoss {
    var loss: Double = None
    var y: Tensor<Double>?
    var t: Tensor<Double>?
    
    func forward(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        self.t = t
        let y = softmax(x)
        self.y = y
        loss = crossEntropyError(y: y, t: t)
        return loss
    }
    
    func backward(dout: Double = 1) -> Tensor<Double> {
        guard let y = y,  let t = t else { return Tensor<Double>(dimensions: [1]) }
        let batch_size = Double(t.elements.count)
        let dx = (y.elements - t.elements) / batch_size
        return Tensor<Double>(Matrix(rows: y.dimensions[0], columns: y.dimensions[1], elements:dx))
    }
}

