//
//  ViewController.swift
//  testDeepLearning
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import UIKit
import Upsurge

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        testGradient()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func getTrainBatch(_ x_train: Tensor<Double>, _ t_train: Tensor<Double>, _ batchSize: Int ) -> (ValueArray<Double>, ValueArray<Double>) {
        var array = Array<Int>()
        for _ in 0..<batchSize {
            array.append(Int(arc4random_uniform(UInt32(x_train.elements.count / 784))))
        }
        
        let x_batch = ValueArray<Double>(capacity: 784 * batchSize)
        for i in 0..<batchSize {
            for k in 0..<784 {
                x_batch.append(x_train.elements[array[i] * 784 + k])
            }
        }
        
        let t_batch = ValueArray<Double>(capacity: 10 * batchSize)
        for i in 0..<batchSize {
            for k in 0..<10 {
                t_batch.append(t_train.elements[array[i] * 10 + k])
            }
        }
        
        return (x_batch, t_batch)
    }
    
    func getTestBatch(_ x_test: Tensor<Double>, _ t_test: Tensor<Double>, _ batchSize: Int ) -> (ValueArray<Double>, ValueArray<Double>) {
        var array = Array<Int>()
        for _ in 0..<batchSize {
            array.append(Int(arc4random_uniform(UInt32(x_test.elements.count / 784))))
        }
        
        let x_test_batch = ValueArray<Double>(capacity: 784 * batchSize)
        for i in 0..<batchSize {
            for k in 0..<784 {
                x_test_batch.append(x_test.elements[array[i] * 784 + k])
            }
        }
        
        let t_test_batch = ValueArray<Double>(capacity: 10 * batchSize)
        for i in 0..<batchSize {
            for k in 0..<10 {
                t_test_batch.append(t_test.elements[array[i] * 10 + k])
            }
        }
        
        return (x_test_batch, t_test_batch)
    }
    
    func testGradient() {
        debugPrint("start..")
        let ((x_train, t_train), (x_test, t_test)) = load_mnist(flatten: true, normalize: true, one_hot_label: true)
        let network = Network2(inputSize: 784, hiddeSize: 50, outputSize: 10)
        
        var trainLossList = Array<Double>()
        let itersNum = 10000
        let batchSize = 100
        let trainSize = x_train.dimensions[0]
        let learningRate: Double = 0.1
        let iterPerEpoch = max(trainSize / batchSize, 1)
        
        for i in 0..<itersNum {
            
            let (x_batch, t_batch) = getTrainBatch(x_train, t_train, batchSize)
            let (x_test_batch, t_test_batch) = getTestBatch(x_test, t_test, batchSize)
            ///
            let tensorXBatch = Tensor<Double>(Matrix<Double>(rows: batchSize, columns: 784, elements: x_batch))
            let tensorTBatch = Tensor<Double>(Matrix<Double>(rows: batchSize, columns: 10, elements: t_batch))
            let tensorXTestBatch = Tensor<Double>(Matrix<Double>(rows: batchSize, columns: 784, elements: x_test_batch))
            let tensorTTestBatch = Tensor<Double>(Matrix<Double>(rows: batchSize, columns: 10, elements: t_test_batch))
            
            //let grad = network.numerical_gradient(x: tensorXBatch, t: tensorTBatch)
            let grad = network.gradient(x: tensorXBatch, t: tensorTBatch)
            
            for key in grad.keys {
                if let item = network.params[key] {
                    network.params[key] = Tensor<Double>((item.elements - (learningRate * (grad[key] as! Tensor<Double>).elements)).toMatrix(rows: item.dimensions[0], columns: item.dimensions[1]))
                }
            }
            
            let loss = network.loss(x: tensorXBatch, t: tensorTBatch)
            trainLossList.append(loss)
            //debugPrint("loss: \(loss)")
            
            if 0 == i % iterPerEpoch {
                let trainAcc = network.accuracy(x: tensorXBatch, t: tensorTBatch)
                let testAcc = network.accuracy(x: tensorXTestBatch, t: tensorTTestBatch)
                //                trainAccList.append(trainAcc)
                //                testAccList.append(testAcc)
                debugPrint("\(trainAcc), \(testAcc)")
            }
        }
        
        debugPrint("end.")
    }
}

