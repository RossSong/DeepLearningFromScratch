//
//  SimpleNet.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 12. 31..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

class simpleNet {
    var W = MyRandomGenerator.randn(inputSize: 2, outputSize: 3)

    func predict(x: Tensor<Double>) -> Tensor<Double> {
        let x = Matrix<Double>(rows:x.dimensions[0], columns:x.dimensions[1], elements: x.elements)
        let w = Matrix<Double>(rows:W.dimensions[0], columns:W.dimensions[1], elements: W.elements)
        return Tensor<Double>(x * w)
    }

    func loss(x: Tensor<Double>, t: Tensor<Double>) -> Double {
        let z = self.predict(x: x)
        let y = softmax(z)
        let loss = crossEntropyError(y: y, t: t)
        return loss
    }
    
}
