package com.hunag.dmoz;

import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.Problem;

public class DeepClassificationTest {

	@Test
	public void test() throws IOException {
		String trainfile = "F:\\DataSets\\dmoz2010\\train.txt";
		String validfile = "F:\\DataSets\\dmoz2010\\validation.txt";
		int[] topn = {5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500};
		Problem train = FileIO.readProblem(trainfile, -1);
		Problem valid = FileIO.readProblem(validfile, -1);
		DeepClassification dc = new DeepClassification(train, 10);
		dc.getClassCenter();
//		for (int i = 0; i < topn.length; i++) {
//			int tn = topn[i];
//			dc.setTopn(tn);
//			dc.predict(valid);
//		}
		String trainclass = "trainStatistic.txt";
		String validclass = "validStatistic.txt";
		dc.statisticNumOfeachClass(train, trainclass);
		dc.statisticNumOfeachClass(valid, validclass);
		dc.setTopn(10);
		dc.transformSamples(valid, false);
		System.out.println(valid.l);
	}
}
