//
//  Gate.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

class Gate {
    func oldAND(x1: Float, x2: Float) -> Float {
        let w1: Float = 0.5
        let w2: Float = 0.5
        let theta: Float = 0.7
        let tmp = x1 * w1 + x2 * w2
        
        if tmp <= theta {
            return 0
        } else {
            return 1
        }
    }
    
    func AND(x1: Double, x2: Double) -> Double {
        let x: [Double] = [x1, x2]
        let w: [Double] = [0.5, 0.5]
        let b = -0.7
        
        let tmp = Upsurge.sum( w * x ) + b
        
        if tmp <= 0 {
            return 0
        } else {
            return 1
        }
    }
    
    func NAND(x1: Double, x2: Double) -> Double {
        let x: [Double] = [x1, x2]
        let w: [Double] = [-0.5, -0.5]
        let b = 0.7
        
        let tmp = Upsurge.sum( w * x ) + b
        
        if tmp <= 0 {
            return 0
        } else {
            return 1
        }
    }
    
    func OR(x1: Double, x2: Double) -> Double {
        let x: [Double] = [x1, x2]
        let w: [Double] = [0.5, 0.5]
        let b = -0.2
        
        let tmp = Upsurge.sum( w * x ) + b
        
        if tmp <= 0 {
            return 0
        } else {
            return 1
        }
    }
    
    func XOR(x1: Double, x2: Double) -> Double {
        let s1 = NAND(x1: x1, x2: x2)
        let s2 = OR(x1: x1, x2: x2)
        let y = AND(x1: s1, x2: s2)
        return y
    }
}
