//
//  MulLayer.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 12. 24..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

class MulLayer {
    var mX: Double = 0.0
    var mY: Double = 0.0
    
    func forward(x: Double, y: Double) -> Double {
        mX = x
        mY = y
        let out = x * y
        
        return out
    }
    
    func backward(dout: Double) -> (Double, Double){
        let dx = dout * mY
        let dy = dout * mX
        
        return (dx, dy)
    }
}
