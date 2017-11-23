//
//  MNIST.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 11. 23..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func load_mnist(flatten: Bool, normalize: Bool) -> ((TwoDimensionalTensorSlice<Double>, TwoDimensionalTensorSlice<Double>), (TwoDimensionalTensorSlice<Double>, TwoDimensionalTensorSlice<Double>)) {
    let data = GetMNISTData()
    
    let rows = data.images.count / (28 * 28)
    let train_rows = (rows)/100 * 80
    
    let tensorX = Tensor<Double>(dimensions: [rows, (28 * 28)])
    
    for i in 0..<rows {
        for j in 0..<(28 * 28) {
            tensorX[i,j] = Double(data.images[i * 28 + j])
        }
    }
    
    let tensorY = Tensor<Double>(dimensions: [rows])
    for i in 0..<rows {
        tensorY[i] = Double(data.labels[i])
    }
    
    //return ((x_train, t_train), (x_test, t_test))
    return ((tensorX.asMatrix(0...(train_rows-1), 0...(28*28 - 1)),
             (tensorY.asMatrix(0...(train_rows-1), 0...0))),
             (tensorX.asMatrix(train_rows...(rows-1), 0...(28*28 - 1)),
              tensorY.asMatrix(train_rows...(rows-1), 0...0)))
    
}


