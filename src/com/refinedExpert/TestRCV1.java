package com.refinedExpert;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class TestRCV1 {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\RCV1RCV2\\vectors\\lyrl2004_vectors_train.dat";
		String qls = "F:\\DataSets\\RCV1RCV2\\rcv1-v2.topics.qrels";
		String topics = "F:\\DataSets\\RCV1RCV2\\rcv1.topics.txt";
		
		String test0 = "F:\\DataSets\\RCV1RCV2\\vectors\\lyrl2004_vectors_test_pt0.dat";
		String test1 = "F:\\DataSets\\RCV1RCV2\\vectors\\lyrl2004_vectors_test_pt1.dat";
		String test2 = "F:\\DataSets\\RCV1RCV2\\vectors\\lyrl2004_vectors_test_pt2.dat";
		String test3 = "F:\\DataSets\\RCV1RCV2\\vectors\\lyrl2004_vectors_test_pt3.dat";
		
		Map<Integer, int[]> id2label = FileInputOutput.getIDtoLabel(qls, topics);
		Problem train = Problem.readProblem(new File(trainfile), 1);
		train.y = FileInputOutput.getLabel(train.y, id2label);
		
		double[] k = {1, 3, 5, 7, 11, 15, 21, 31, 41, 51};
		double[] c = new double[15];
		for (int i = 0; i < c.length; i++) {
			c[i] = Math.pow(2, i - 7);
		}
		RefinedExpert re = null;
		for (int i = 0; i < c.length; i++) {
			Parameter param = new Parameter(c[i], 1000, 0.001);
			re = new RefinedExpert(train, param, k, 10, RefinedExpert.MICROF1);
			re.train();
		}
		
		Problem p0 = Problem.readProblem(new File(test0), 1);
		Problem p1 = Problem.readProblem(new File(test1), 1);
		Problem p2 = Problem.readProblem(new File(test2), 1);
		Problem p3 = Problem.readProblem(new File(test3), 1);
		
		int l = p0.l + p1.l + p2.l + p3.l;
		int n = Math.max(p0.n, Math.max(p1.n, Math.max(p2.n, p3.n)));
		Problem p = new Problem();
		p.l = l;
		p.n = n;
		p.x = new DataPoint[l][];
		p.y = new int[l][];
		int index = 0;
		for (int i = 0; i < p0.l; i++) {
			p.x[index] = p0.x[i];
			p.y[index] = p0.y[i];
			index++;
		}
		for (int i = 0; i < p1.l; i++) {
			p.x[index] = p1.x[i];
			p.y[index] = p1.y[i];
			index++;
		}
		for (int i = 0; i < p2.l; i++) {
			p.x[index] = p2.x[i];
			p.y[index] = p2.y[i];
			index++;
		}
		for (int i = 0; i < p3.l; i++) {
			p.x[index] = p3.x[i];
			p.y[index] = p3.y[i];
			index++;
		}
		p.y = FileInputOutput.getLabel(p.y, id2label);
		double[] ks = {3};
		Parameter param = new Parameter(1, 1000, 0.001);
		RefinedExpert res = new RefinedExpert(train, param, ks, 10, RefinedExpert.MICROF1);
		res.trainAndTest(p);
	}

}
