//
//  neuron.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

class Network {
    var network: Dictionary<String, Any>!
    init() {
        network = Dictionary<String, Any>()
        network["W1"] = Matrix([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network["b1"] = Matrix([[0.1, 0.2, 0.3]])
        network["W2"] = Matrix([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network["b2"] = Matrix([[0.1, 0.2]])
        network["W3"] = Matrix([[0.1, 0.3], [0.2, 0.4]])
        network["b3"] = Matrix([[0.1, 0.2]])
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
            result[i] = Upsurge.dot(x, newY[i])
        }
        
        return result
    }
    
    func identity_function(_ x:Matrix<Double>) -> Matrix<Double> {
        return x
    }
    
    func forword(_ x: [Double]) -> [Double] {
        let matX:Matrix = Matrix([x])
        let W1:Matrix = network["W1"] as! Matrix<Double>
        let W2:Matrix = network["W2"] as! Matrix<Double>
        let W3:Matrix = network["W3"] as! Matrix<Double>
        
        let b1:Matrix = network["b1"] as! Matrix<Double>
        let b2:Matrix = network["b2"] as! Matrix<Double>
        //let b3:[Double] = network["b3"] as! [Double]
        
        let a1 = matX * W1  + b1
        let z1 = sigmoid(a1)
        
        let a2 = z1 * W2 + b2
        let z2 = sigmoid(a2)

        let a3 = z2 * W3 + b2
        let y = identity_function(a3)
        
        let row = y.row(0)
        var result = Array<Double>(repeating: 0.0, count: row.count)
        for i in 0..<row.count {
            result[i] = row[i]
        }
       
        return result
    }
}
