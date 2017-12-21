//
//  MNIST.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 11. 23..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func load_mnist(flatten: Bool, normalize: Bool) -> ((Tensor<Double>, Tensor<Double>), (Tensor<Double>, Tensor<Double>)) {
    let data = GetMNISTData()
    
    let rows = data.images.count / (28 * 28)
    let train_rows = (rows)/100 * 80
    
    let tensorX = Tensor<Double>(dimensions: [train_rows, (28 * 28)])
    let tensorTestX = Tensor<Double>(dimensions: [rows - train_rows, (28 * 28)])
    
    for i in 0..<rows {
        if i < train_rows {
            for j in 0..<(28 * 28) {
                tensorX[i,j] = Double(data.images[i * (28*28) + j])
            }
        }
        else {
            for j in 0..<(28 * 28) {
                tensorTestX[i - train_rows,j] = Double(data.images[i * (28*28) + j])
            }
        }
    }
    
    let tensorY = Tensor<Double>(dimensions: [train_rows])
    let tensorTestY = Tensor<Double>(dimensions: [rows - train_rows])
    
    for i in 0..<rows {
        if i < train_rows {
            tensorY[i] = Double(data.labels[i])
        }
        else {
            tensorTestY[i - train_rows] = Double(data.labels[i])
        }
    }
    
    
    //return ((x_train, t_train), (x_test, t_test))
    return ((tensorX, tensorY), (tensorTestX, tensorTestY))
    
}


