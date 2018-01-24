package com.scene;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.tools.FileIO;

public class TestTransformFormat {

	@Test
	public void test() throws IOException {
		String trainarff = "F:\\DataSets\\scene\\scene-train.arff";
		String testarff = "F:\\DataSets\\scene\\scene-test.arff";
		String trainsvm = "F:\\DataSets\\scene\\scene-train.svm";
		String testsvm = "F:\\DataSets\\scene\\scene-test.svm";
		
		FileIO.transArffToSVM(trainarff, trainsvm, 294);
		FileIO.transArffToSVM(testarff, testsvm, 294);
	}

}
