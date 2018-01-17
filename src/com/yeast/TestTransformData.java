package com.yeast;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.tools.FileIO;

public class TestTransformData {

	@Test
	public void test() throws IOException {
		String arffFile = "F:\\DataSets\\yeast\\yeast\\yeast-train.arff";
		String svmfile = "F:\\DataSets\\yeast\\yeast\\yeast-train.svm";
		FileIO.transArffToSVM(arffFile, svmfile, 103);
	}

}
