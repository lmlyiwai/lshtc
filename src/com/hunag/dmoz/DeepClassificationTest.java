package com.hunag.dmoz;

import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.Problem;

public class DeepClassificationTest {

	@Test
	public void test() throws IOException {
		String trainfile = "F:\\DataSets\\dmoz2010\\train.txt";
		Problem train = FileIO.readProblem(trainfile, -1);
		DeepClassification dc = new DeepClassification(train, 10);
		dc.getClassCenter();
		System.out.println();
	}

}
