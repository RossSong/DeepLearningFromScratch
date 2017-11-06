//
//  loss.swift
//  testDeepLearning
//
//  Created by Ross on 2017. 11. 1..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func mean_squared_error(y: [Double], t:[Double]) -> Double {
    let _y = ValueArray<Double>(y)
    let _t = ValueArray<Double>(t)
    let _s = _y - _t
    let _sq = _s * _s
    let sum = Upsurge.sum(_sq)
    return 0.5 * sum
}
