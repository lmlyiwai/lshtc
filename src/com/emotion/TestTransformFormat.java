package com.emotion;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.tools.FileIO;

public class TestTransformFormat {

	@Test
	public void test() throws IOException {
		String trainarff = "F:\\DataSets\\emotion\\emotions-train.arff";
		String testarff = "F:\\DataSets\\emotion\\emotions-test.arff";
		String trainsvm = "F:\\DataSets\\emotion\\emotions-train.svm";
		String testsvm = "F:\\DataSets\\emotion\\emotions-test.svm";
		
		FileIO.transArffToSVM(trainarff, trainsvm, 72);
		FileIO.transArffToSVM(testarff, testsvm, 72);
	}

}
