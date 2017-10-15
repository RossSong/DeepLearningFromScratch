//
//  neuron.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Surge

class Network {
    var network: Dictionary<String, Any>!
    init() {
        network = Dictionary<String, Any>()
        network["W1"] = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
        network["b1"] = [0.1, 0.2, 0.3]
        network["W2"] = [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
        network["b2"] = [0.1, 0.2]
        network["W3"] = [[0.1, 0.3], [0.2, 0.4]]
        network["b3"] = [0.1, 0.2]
    }
    
    func dot(x:[Double], y:[[Double]]) -> [Double] {
        var result = Array<Double>(repeating: 0.0, count: y[0].count)
        var newY = Array<Array<Double>>(repeating: Array<Double>(repeating: 0.0, count: y.count), count: y[0].count)
        
        for i in 0..<newY.count {
            for j in 0..<newY[i].count {
                newY[i][j] = y[j][i]
            }
        }
        
        for i in 0..<newY.count {
            result[i] = Surge.dot(x, y: newY[i])
        }
        
        return result
    }
    
    func identity_function(_ x:[Double]) -> [Double] {
        return x
    }
    
    func forword(_ x: [Double]) -> [Double] {
        let W1:[[Double]] = network["W1"] as! [[Double]]
        let W2:[[Double]] = network["W2"] as! [[Double]]
        let W3:[[Double]] = network["W3"] as! [[Double]]
        
        let b1:[Double] = network["b1"] as! [Double]
        let b2:[Double] = network["b2"] as! [Double]
        //let b3:[Double] = network["b3"] as! [Double]
        
        let a1 = dot(x:x, y:W1) + b1
        let z1 = sigmoid(a1)
        
        let a2 = dot(x:z1, y:W2) + b2
        let z2 = sigmoid(a2)
        
        let a3 = dot(x:z2, y:W3) + b2
        let y = identity_function(a3)
        
        return y
    }
}
