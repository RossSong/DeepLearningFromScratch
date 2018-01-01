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
    var indexes: [Int]?
    
    func makeIndexes(_ x: Tensor<Double>) {
        self.indexes = Array<Int>()
        for i in 0..<x.elements.count {
            if x.elements[i] < 0 {
                indexes?.append(i)
            }
        }
    }
    
    func setupZefoForIndexes(_ x: Tensor<Double>) {
        guard let indexes = indexes else { return }
        for i in 0..<indexes.count {
            x.elements[i] = 0
        }
    }
    
    func forward(x: Tensor<Double>) -> Tensor<Double> {
        let out = x.copy()
        
        makeIndexes(x)
        setupZefoForIndexes(out)
        
        return out
    }
    
    func backward(dout: Tensor<Double>) -> Tensor<Double> {
        let out = dout.copy()
        setupZefoForIndexes(out)
        return out
    }
}

