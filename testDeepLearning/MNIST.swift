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

func setupTrainLabelData(labels: Array<UInt8>, tensorY: Tensor<Double>, i: Int, j: Int) {
    tensorY[i,j] = 0
    if j == labels[i] {
        tensorY[i,j] = 1
    }
}

func setupTestLabelData(labels: Array<UInt8>, train_rows: Int, tensorTestY: Tensor<Double>, i: Int, j: Int) {
    tensorTestY[i - train_rows,j] = 0
    if j == labels[i] {
        tensorTestY[i - train_rows,j] = 1
    }
}

func doSetupOneHotLabels(labels: Array<UInt8>, train_rows: Int, tensorY: Tensor<Double>, tensorTestY: Tensor<Double>, i: Int, j: Int) {
    if i < train_rows {
        setupTrainLabelData(labels: labels, tensorY: tensorY, i: i, j: j)
    }
    else {
        setupTestLabelData(labels: labels, train_rows: train_rows, tensorTestY: tensorTestY, i: i, j: j)
    }
}

func setupOneHotLabels(labels: Array<UInt8>, rows: Int, train_rows: Int, tensorY: Tensor<Double>, tensorTestY: Tensor<Double>) {
    for i in 0..<rows {
        for j in 0..<10 {
            doSetupOneHotLabels(labels: labels, train_rows: train_rows, tensorY: tensorY, tensorTestY: tensorTestY, i: i, j: j)
        }
    }
}

func setupNoOneHotLabels(labels: Array<UInt8>, rows: Int, train_rows: Int, tensorY: Tensor<Double>, tensorTestY: Tensor<Double>) {
    for i in 0..<rows {
        if i < train_rows {
            tensorY[i] = Double(labels[i])
        }
        else {
            tensorTestY[i - train_rows] = Double(labels[i])
        }
    }
}

func setupY(one_hot_label: Bool, labels: Array<UInt8>, rows: Int, train_rows: Int, tensorY: Tensor<Double>, tensorTestY: Tensor<Double>) {
    if one_hot_label {
        setupOneHotLabels(labels: labels, rows: rows, train_rows: train_rows, tensorY: tensorY, tensorTestY: tensorTestY)
    }
    else {
        setupNoOneHotLabels(labels: labels, rows: rows, train_rows: train_rows, tensorY: tensorY, tensorTestY: tensorTestY)
    }
}

func saveImageDataFile(_ dataMNIST: GetMNISTData) {
    let pointerImages = UnsafeBufferPointer(start:dataMNIST.images, count:dataMNIST.images.count)
    let dataImages = Data(buffer:pointerImages)
    saveMNISTData(data: dataImages, file: "MNIST-Images.data")
}

func saveLabelDataFile(_ dataMNIST: GetMNISTData) {
    let pointerLabels = UnsafeBufferPointer(start:dataMNIST.labels, count:dataMNIST.labels.count)
    let dataLabels = Data(buffer:pointerLabels)
    saveMNISTData(data: dataLabels, file: "MNIST-Labels.data")
}

func saveLoadAndPreparedDataFile() {
    let dataMNIST = GetMNISTData()
    saveImageDataFile(dataMNIST)
    saveLabelDataFile(dataMNIST)
}

func getDimensionForLabels(one_hot_label: Bool, rows: Int, train_rows: Int) -> ([Int], [Int]) {
    var dimensionTrain = [train_rows]
    var dimensionTest = [rows - train_rows]
    
    if one_hot_label {
        dimensionTrain = [train_rows, 10]
        dimensionTest = [rows - train_rows, 10]
    }
    
    return (dimensionTrain, dimensionTest)
}

func getDelta(normalize: Bool) -> Double {
    guard normalize else { return 1.0 }
    return 255.0
}

func setupImageData(images: [UInt8], normalize: Bool, i: Int, tensorX: Tensor<Double>) {
    let delta = getDelta(normalize: normalize)
    
    for j in 0..<(28 * 28) {
        tensorX[i,j] = Double(images[i * (28*28) + j]) / delta
    }
}

func setupImageDataForTest(images: [UInt8], normalize: Bool, train_rows: Int, i: Int, tensorTestX: Tensor<Double>) {
    let delta = getDelta(normalize: normalize)
    
    for j in 0..<(28 * 28) {
        tensorTestX[i - train_rows,j] = Double(images[i * (28*28) + j]) / delta
    }
}

func setupX(images: [UInt8], normalize: Bool, rows: Int, train_rows: Int, tensorX: Tensor<Double>, tensorTestX: Tensor<Double>) {
    for i in 0..<rows {
        if i < train_rows {
            setupImageData(images: images, normalize: normalize, i: i, tensorX: tensorX)
        }
        else {
            setupImageDataForTest(images: images, normalize: normalize, train_rows: train_rows, i: i, tensorTestX: tensorTestX)
        }
    }
}

func getXAndTestX(images: [UInt8], normalize: Bool, rows: Int, train_rows: Int) -> (Tensor<Double>, Tensor<Double>) {
    let tensorX = Tensor<Double>(dimensions: [train_rows, (28 * 28)])
    let tensorTestX = Tensor<Double>(dimensions: [rows - train_rows, (28 * 28)])
    setupX(images: images, normalize: normalize, rows: rows, train_rows: train_rows, tensorX: tensorX, tensorTestX: tensorTestX)
    return (tensorX, tensorTestX)
}

func getYAndTestY(labels: [UInt8], normalize: Bool, one_hot_label: Bool, rows: Int, train_rows: Int) -> (Tensor<Double>, Tensor<Double>) {
    let (dimensionTrain, dimensionTest) = getDimensionForLabels(one_hot_label: one_hot_label, rows: rows, train_rows: train_rows)
    let tensorY = Tensor<Double>(dimensions: dimensionTrain)
    let tensorTestY = Tensor<Double>(dimensions: dimensionTest)
    
    setupY(one_hot_label: one_hot_label, labels: labels, rows: rows, train_rows: train_rows, tensorY: tensorY, tensorTestY: tensorTestY)
    return (tensorY, tensorTestY)
}

func load_mnist(flatten: Bool, normalize: Bool, one_hot_label: Bool = false) -> ((Tensor<Double>, Tensor<Double>), (Tensor<Double>, Tensor<Double>)) {
    
    var images: [UInt8] = loadMNISTData(file: "MNIST-Images.data")
    var labels: [UInt8] = loadMNISTData(file: "MNIST-Labels.data")
    
    if 0 == images.count || 0 == images.count {
        saveLoadAndPreparedDataFile()
        images = loadMNISTData(file: "MNIST-Images.data")
        labels = loadMNISTData(file: "MNIST-Labels.data")
    }
    
    let rows = images.count / (28 * 28)
    let train_rows = (rows)/100 * 80
    let (tensorX, tensorTestX) = getXAndTestX(images: images, normalize: normalize, rows: rows, train_rows: train_rows)
    let (tensorY, tensorTestY) = getYAndTestY(labels: labels, normalize: normalize, one_hot_label: one_hot_label, rows: rows, train_rows: train_rows)
    
    //return ((x_train, t_train), (x_test, t_test))
    return ((tensorX, tensorY), (tensorTestX, tensorTestY))
    
}
