//
//  MNIST.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 11. 23..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import Foundation
import Upsurge

func saveMNISTData(data: Data, file: String) {
    guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
    let fileURL = dir.appendingPathComponent(file)
    
    do {
        try data.write(to: fileURL)
    }
    catch {
        debugPrint("failed to save: \(error)")
    }
}

func loadMNISTData(file: String) -> [UInt8] {
    guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return [UInt8]() }
    let fileURL = dir.appendingPathComponent(file)
        
    guard let data = NSData(contentsOf: fileURL) else { return [UInt8]() }
    var bytes = [UInt8]()
    var buffer = [UInt8](repeating: 0, count: data.length)
    data.getBytes(&buffer, length: data.length)
    bytes = buffer
    
    return bytes
}

func checkFileExist(file: String) -> Bool {
    let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] as String
    let url = NSURL(fileURLWithPath: path)
    if let pathComponent = url.appendingPathComponent(file) {
        let filePath = pathComponent.path
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: filePath) {
            return true
        }
    }
    
    return false
}

func load_mnist(flatten: Bool, normalize: Bool) -> ((Tensor<Double>, Tensor<Double>), (Tensor<Double>, Tensor<Double>)) {
    
    var images: [UInt8] = loadMNISTData(file: "MNIST-Images.data")
    var labels: [UInt8] = loadMNISTData(file: "MNIST-Labels.data")
    
    if 0 == images.count || 0 == images.count {
        let dataMNIST = GetMNISTData()
        
        let pointerImages = UnsafeBufferPointer(start:dataMNIST.images, count:dataMNIST.images.count)
        let dataImages = Data(buffer:pointerImages)
        saveMNISTData(data: dataImages, file: "MNIST-Images.data")
        
        let pointerLabels = UnsafeBufferPointer(start:dataMNIST.labels, count:dataMNIST.labels.count)
        let dataLabels = Data(buffer:pointerLabels)
        saveMNISTData(data: dataLabels, file: "MNIST-Labels.data")
    }
    
    let rows = images.count / (28 * 28)
    let train_rows = (rows)/100 * 80
    
    let tensorX = Tensor<Double>(dimensions: [train_rows, (28 * 28)])
    let tensorTestX = Tensor<Double>(dimensions: [rows - train_rows, (28 * 28)])
    
    for i in 0..<rows {
        if i < train_rows {
            for j in 0..<(28 * 28) {
                tensorX[i,j] = Double(images[i * (28*28) + j])
            }
        }
        else {
            for j in 0..<(28 * 28) {
                tensorTestX[i - train_rows,j] = Double(images[i * (28*28) + j])
            }
        }
    }
    
    let tensorY = Tensor<Double>(dimensions: [train_rows])
    let tensorTestY = Tensor<Double>(dimensions: [rows - train_rows])
    
    for i in 0..<rows {
        if i < train_rows {
            tensorY[i] = Double(labels[i])
        }
        else {
            tensorTestY[i - train_rows] = Double(labels[i])
        }
    }
    
    
    //return ((x_train, t_train), (x_test, t_test))
    return ((tensorX, tensorY), (tensorTestX, tensorTestY))
    
}
