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
        let y = softmax(a)
        XCTAssert(y[0] == 0.018211273295547531)
        XCTAssert(y[1] == 0.24519181293507392)
        XCTAssert(y[2] == 0.73659691376937864)
    }
    
    func testSoftMax2() {
        let a = ValueArray<Double>([1010, 1000, 990])
        let y = softmax(a)
        XCTAssert(y[0] == 0.99995460007033099)
        XCTAssert(y[1] == 4.5397868608866649e-05)
        XCTAssert(y[2] == 2.0610600462090622e-09)
    }
    
    func testLoadMNIST() {
        let tensor = Tensor<Double>(dimensions:[(28*28)], repeatedValue: 0.0)
        let k = tensor.asMatrix(0...(28*28-1),0...0)
        let ((x,y), (tx,ty)) = load_mnist(flatten: true, normalize: false)
        let a = x * k
        let b = tx * k
        XCTAssert(8000 == a.rows)
        XCTAssert(1 == a.columns)
        XCTAssert(2000 == b.rows)
        XCTAssert(1 == b.columns)
    }
    
    func testCrossEntropyError() {
        let ret = crossEntropyError(y: Tensor<Double>(Matrix<Double>([[1.0,2.0,3.0], [1.0,2.0,3.0]])), t:Tensor<Double>(Matrix<Double>([[1.0,3.0,2.0], [1.0,3.0,2.0]])))
        debugPrint(ret)
        XCTAssert(-4.2766661190160544 == ret)
    }
}
