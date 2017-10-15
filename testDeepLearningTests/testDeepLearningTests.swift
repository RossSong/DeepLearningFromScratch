//
//  testDeepLearningTests.swift
//  testDeepLearningTests
//
//  Created by RossSong on 2017. 10. 15..
//  Copyright © 2017년 RossSong. All rights reserved.
//

import XCTest
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
}
