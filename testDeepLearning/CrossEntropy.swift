//
//  CrossEntropy.swift
//  testDeepLearning
//
//  Created by RossSong on 2018. 1. 1..
//  Copyright © 2018년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func crossEntropyError(y: Tensor<Double>, t: Tensor<Double>) -> Double {
    let batch_size: Double = Double(y.dimensions[0])
    if y.elements.count != t.elements.count || 0 == batch_size {
        debugPrint("Error - crossEntropyError")
    }
    
    let val = Upsurge.sum(t.elements * Upsurge.log(y.elements))
    if 0 == val {
        return 0
    }
    
    return -1 * val / batch_size
}

