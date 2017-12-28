//
//  testDeepLearningTests.swift
//  testDeepLearningTests
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import XCTest
import Upsurge
@testable import testDeepLearning

class testDeepLearningTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testAND() {
        let gate = Gate()
        XCTAssertTrue(0.0 == gate.AND(x1: 0.0, x2: 0.0))
        XCTAssertTrue(0.0 == gate.AND(x1: 1.0, x2: 0.0))
        XCTAssertTrue(0.0 == gate.AND(x1: 0.0, x2: 1.0))
        XCTAssertTrue(1.0 == gate.AND(x1: 1.0, x2: 1.0))
    }
    
    func testNAND() {
        let gate = Gate()
        XCTAssertTrue(1.0 == gate.NAND(x1: 0.0, x2: 0.0))
        XCTAssertTrue(1.0 == gate.NAND(x1: 1.0, x2: 0.0))
        XCTAssertTrue(1.0 == gate.NAND(x1: 0.0, x2: 1.0))
        XCTAssertTrue(0.0 == gate.NAND(x1: 1.0, x2: 1.0))
    }
    
    func testOR() {
        let gate = Gate()
        XCTAssertTrue(0.0 == gate.OR(x1: 0.0, x2: 0.0))
        XCTAssertTrue(1.0 == gate.OR(x1: 1.0, x2: 0.0))
        XCTAssertTrue(1.0 == gate.OR(x1: 0.0, x2: 1.0))
        XCTAssertTrue(1.0 == gate.OR(x1: 1.0, x2: 1.0))
    }
    
    func testXOR() {
        let gate = Gate()
        XCTAssertTrue(0.0 == gate.XOR(x1: 0.0, x2: 0.0))
        XCTAssertTrue(1.0 == gate.XOR(x1: 1.0, x2: 0.0))
        XCTAssertTrue(1.0 == gate.XOR(x1: 0.0, x2: 1.0))
        XCTAssertTrue(0.0 == gate.XOR(x1: 1.0, x2: 1.0))
    }
    
    func testNetwork() {
        let network = Network()
        let x = [1.0, 0.5]
        var y = network.forword(x)
    
        XCTAssertTrue(0.31682708 == Float(y[0])) //0.31682707641102981
        XCTAssertTrue(0.69627909 == Float(y[1])) //0.69627908986196685
    }
    
    func test_mean_squared_error() {
        let t:[Double] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        let y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
        let ret = mean_squared_error(y: y, t: t)
        debugPrint(ret)
        XCTAssert(0.097500000000000031 == ret)
    }
    
    func test_mean_squared_error2() {
        let t:[Double] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        let y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
        let ret = mean_squared_error(y: y, t: t)
        debugPrint(ret)
        XCTAssert(0.59750000000000003 == ret)
    }
    
    func testSoftMax() {
        let a = ValueArray<Double>([0.3, 2.9, 4.0])
        let y = softmax(Tensor<Double>(a.toRowMatrix()))
        XCTAssert(y[0] == 0.018211273295547531)
        XCTAssert(y[1] == 0.24519181293507392)
        XCTAssert(y[2] == 0.73659691376937864)
    }
    
    func testSoftMax2() {
        let y = softmax(Tensor<Double>(Matrix<Double>([[1010], [1000],[990]])))
        XCTAssert(y[0, 0] == 0.99995460007033099)
        XCTAssert(y[1, 0] == 4.5397868608866649e-05)
        XCTAssert(y[2, 0] == 2.0610600462090622e-09)
    }
    
//    func testLoadMNIST() {
//        let tensor = Tensor<Double>(dimensions:[(28*28)], repeatedValue: 0.0)
//        let k = tensor.elements.toColumnMatrix()
//        debugPrint("rows:\(k.rows), cols:\(k.columns)")
//
//        let ((x,y), (tx,ty)) = load_mnist(flatten: true, normalize: false)
//
//        let tA = x.asMatrix(0...((x.elements.count/(28*28)) - 1), 0...((28*28)-1))
//        let a =  tA * k
//        let tB = tx.asMatrix(0...1999, 0...783)
//        let b = tB * k
//        XCTAssert(8000 == a.rows)
//        XCTAssert(1 == a.columns)
//        XCTAssert(2000 == b.rows)
//        XCTAssert(1 == b.columns)
//    }
    
    func testCrossEntropyError() {
        let ret = crossEntropyError(y: Tensor<Double>(Matrix<Double>([[1.0,2.0,3.0], [1.0,2.0,3.0]])), t:Tensor<Double>(Matrix<Double>([[1.0,3.0,2.0], [1.0,3.0,2.0]])))
        debugPrint(ret)
        XCTAssert(-4.2766661190160544 == ret)
    }
    
    func testCrossEntropyError2() {
        let ret = crossEntropyError(y: Tensor<Double>(Matrix<Double>([[1.0,2.0,3.0]])), t:Tensor<Double>(Matrix<Double>([[1.0,4.0,5.0]])))
        debugPrint(ret)
        XCTAssert(-8.2656501655803289 == ret)
    }
    
    func testReluForword() {
        let x = Tensor<Double>(Matrix([[-1, -2, 3]]))
        let y = Relu().forward(x: x)
        XCTAssert(0 == y[0,0])
        XCTAssert(0 == y[0,1])
        XCTAssert(3 == y[0,2])
    }
    
    func testReluBackword() {
        let x = Tensor<Double>(Matrix([[-1, -2, 3]]))
        let y = Relu().backward(dout: x)
        XCTAssert(0 == y[0,0])
        XCTAssert(0 == y[0,1])
        XCTAssert(3 == y[0,2])
    }
    
    func loss(W: Tensor<Double>) -> Double {
        let y = Tensor<Double>(Matrix([[0.1, 0.2, 0.7]]))
        let t = Tensor<Double>(Matrix([[0.1, 0.2, 0.7]]))
        return SoftMaxWithLoss().forward(x: y, t: t)
    }
    
    func function_2(x: Tensor<Double>) -> Double {
        let a = (x.elements[0] * x.elements[0])
        let b = (x.elements[1] * x.elements[1])
        return  a + b
    }
    
    func doXCTAssert(_ value: Double,_ target: Double) -> Bool {
        return !(value < target + 0.001 && value > target - 0.001)
    }

    func testNumericalGradient() {
        var x = Tensor<Double>(Matrix([[3.0, 4.0]]))
        var ret = numericalGradient(f: function_2, x: x)
        if doXCTAssert(6.0, ret[0]) { XCTFail() }
        if doXCTAssert(8.0, ret[1]) { XCTFail() }
        
        x = Tensor<Double>(Matrix([[0.0, 2.0]]))
        ret = numericalGradient(f: function_2, x: x)
        if doXCTAssert(0.0, ret[0]) { XCTFail() }
        if doXCTAssert(4.0, ret[1]) { XCTFail() }
        
        x = Tensor<Double>(Matrix([[3.0, 0.0]]))
        ret = numericalGradient(f: function_2, x: x)
        if doXCTAssert(6.0, ret[0]) { XCTFail() }
        if doXCTAssert(0.0, ret[1]) { XCTFail() }
    }
    
    func testMulLayer() {
        let apple: Double = 100
        let apple_num: Double = 2
        let tax: Double = 1.1
        
        let mul_apple_layer = MulLayer()
        let mul_tax_layer = MulLayer()
        
        let apple_price = mul_apple_layer.forward(x: apple, y: apple_num)
        let price = mul_tax_layer.forward(x: apple_price, y: tax)
        
        if doXCTAssert(price, 220) { XCTFail() }
    
        let dPrice: Double = 1.0
        let (dApplePrice, dTax) = mul_tax_layer.backward(dout: dPrice)
        let (dApple, dAppleNum) = mul_apple_layer.backward(dout: dApplePrice)
        if doXCTAssert(dApple, 2.2) { XCTFail() }
        if doXCTAssert(dAppleNum, 110) { XCTFail() }
        if doXCTAssert(dTax, 200) { XCTFail() }
    }
    
    func testCrossEntpropy() {
        let a = Tensor<Double>(Matrix<Double>([[1.17156040198497, -0.909620675200142, 1.39999455813766, 0.0113455066715791, -0.942148990832368, 0.0639886230112339, -0.0075577340029872, 1.93737217777945, -0.895488510669966, -0.274747757767493]]))
        
        let b = Tensor<Double>(Matrix<Double>([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
        
        let s = SoftMaxWithLoss()
        let ret = s.forward(x: a, t: b)
        XCTAssert(3.2324297036275436 == ret)
        
        let retBack = s.backward()
        if doXCTAssert(retBack[0], 0.0167609) { XCTFail() }
        if doXCTAssert(retBack[1], 0.00209147) { XCTFail() }
        if doXCTAssert(retBack[2], 0.02106226) { XCTFail() }
        if doXCTAssert(retBack[3], 0.00525318) { XCTFail() }
        if doXCTAssert(retBack[4], 0.00202453) { XCTFail() }
        if doXCTAssert(retBack[5], 0.00553713) { XCTFail() }
        if doXCTAssert(retBack[6], 0.00515481) { XCTFail() }
        if doXCTAssert(retBack[7], 0.03604833) { XCTFail() }
        if doXCTAssert(retBack[8], 0.00212124) { XCTFail() }
        if doXCTAssert(retBack[9], -0.09605385) { XCTFail() }
    }
    
    
    
    /*
    func testMiniBatch() {
        let ((x_train, t_train), (x_test, t_test)) = load_mnist(flatten: true, normalize: true, one_hot_label: true)
        let network = Network2(inputSize: 784, hiddeSize: 50, outputSize: 10)
        
        var trainLossList = Array<Double>()
        let itersNum = 10000
        let batchSize = 1
        let learningRate: Double = 0.01
        for _ in 0..<itersNum {
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
            
            let tensorXBatch = Tensor<Double>(Matrix<Double>(rows: batchSize, columns: 784, elements: x_batch))
            let tensorTBatch = Tensor<Double>(Matrix<Double>(rows: batchSize, columns: 10, elements: t_batch))
            
            //let grad = network.numerical_gradient(x: tensorXBatch, t: tensorTBatch)
            let grad = network.gradient(x: tensorXBatch, t: tensorTBatch)
            
            for key in grad.keys {
                if let item = network.params[key] {
                    network.params[key] = Tensor<Double>((item.elements - (learningRate * (grad[key] as! Tensor<Double>).elements)).toMatrix(rows: item.dimensions[0], columns: item.dimensions[1]))
                }
            }
            
            let loss = network.loss(x: tensorXBatch, t: tensorTBatch)
            trainLossList.append(loss)
            debugPrint("loss: \(loss)")
        }
    }
    */
    
    /*
    func testNetwork2() {
        let count = 3
        let ((x_train, t_train), (x_test, t_test)) = load_mnist(flatten: true, normalize: true, one_hot_label: true)
        let network = Network2(inputSize: 784, hiddeSize: 50, outputSize: 10)

        let x_batch = ValueArray<Double>(capacity: 784 * count)
        for i in 0..<784 * count {
            x_batch.append(x_train.elements[i])
        }

        let t_batch = ValueArray<Double>(capacity: 10 * count)
        for i in 0..<10 * count {
            t_batch.append(t_train.elements[i])
        }

        let tensorXBatch = Tensor<Double>(Matrix<Double>(rows: count, columns: 784, elements: x_batch))
        let tensorTBatch = Tensor<Double>(Matrix<Double>(rows: count, columns: 10, elements: t_batch))

        let gradNumerical = network.numerical_gradient(x: tensorXBatch, t: tensorTBatch)
        let gradBackProp = network.gradient(x: tensorXBatch, t: tensorTBatch)

        for key in gradNumerical.keys {
            let a = gradBackProp[key] as! Tensor<Double>
            let b = gradNumerical[key] as! Tensor<Double>

            let mA = a.elements
            let mB = b.elements
            let tmp = mA - mB
            let diff = Upsurge.mean(Upsurge.abs(tmp))
            debugPrint("1- \(key): diff \(diff)")
            if doXCTAssert(diff, 0) { XCTFail() }
        }
    }
    
    func testNetwork3() {
        //let ((x_train, t_train), (x_test, t_test)) = load_mnist(flatten: true, normalize: false)
        let network = Network2(inputSize: 2, hiddeSize: 10, outputSize: 1)
        
        let tensorXBatch = Tensor<Double>(Matrix<Double>([[1, 1],
                                                      [0, 1],
                                                      [1, 1]
                                                      ]))
        
        let tensorTBatch = Tensor<Double>(Matrix<Double>([[1],
                                                      [0],
                                                      [1]
            ]))
        
        let gradNumerical = network.numerical_gradient(x: tensorXBatch, t: tensorTBatch)
        let gradBackProp = network.gradient(x: tensorXBatch, t: tensorTBatch)
            
        for key in gradNumerical.keys {
            let a = gradBackProp[key] as! Tensor<Double>
            let b = gradNumerical[key] as! Tensor<Double>
            
            let mA = a.elements
            let mB = b.elements
            let tmp = mA - mB
            let diff = Upsurge.mean(Upsurge.abs(tmp))
            debugPrint("2- \(key) diff : \(diff)")
            if key == "b2" {
               if doXCTAssert(diff, 0.222) { XCTFail() }
            }
            else {
                if doXCTAssert(diff, 0) { XCTFail() }
            }
        }
        
    }*/

}
