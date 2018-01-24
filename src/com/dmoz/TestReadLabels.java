package com.dmoz;

import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.structure.Structure;

public class TestReadLabels {

	@Test
	public void test() throws IOException {
		String labelfile = "F:\\DataSets\\Dmoz\\dmoz_new\\large_lshtc_dataset\\large_lshtc_dataset\\cat_hier.txt";
//		String labelfile = "F:\\DataSets\\Dmoz\\dmoz_new\\large_lshtc_dataset\\large_lshtc_dataset\\test.txt";
		Map<Integer, Integer> map = ReadData.readMap(labelfile);
		Map<Integer, Integer> pam = ReadData.reverseMap(map);
		Structure tree = ReadData.getStructure(labelfile, map);
		Map<Integer, int[]> path = tree.getAllPath();
		int[] leavel = tree.getLeaves();
		for(int i = 0; i < leavel.length; i++) {
			int[] p = path.get(leavel[i]);
			for(int j = 0; j < p.length; j++) {
				System.out.print(pam.get(p[j]) + " ");
			}
		}
		System.out.println();
	}

}
