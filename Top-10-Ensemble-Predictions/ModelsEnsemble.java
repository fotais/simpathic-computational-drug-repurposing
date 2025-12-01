package testproj;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;


public class ModelsEnsemble {
	
	static int t=0;
	static int d=0;

	public static void main(String [] args) {

				
		try {
			//CALCULATE ENSEMBLE of {BoP,COMPGCN, PATHS, ANYBURL} predictions and Query-based metrics
			
			String ddiPath = "/media/fot/USB STICK/FOT/Research/Graph analysis @dimokritos/Drug-mappings github/";
			List<String> queries  =  new LinkedList<String>();
			queries.add("C0011185_C0027126");
			queries.add("C0025887_C0027126");
			queries.add("C3273375_C0027126");
			queries.add("C0001927_C0751882");
			queries.add("C0034261_C0751882");
			queries.add("C0020336_C0043459");
			queries.add("C0008024_C0043459");
			queries.add("C0085994_C0043459");
			queries.add("C0055568_C0043459");
			queries.add("C0042105_C0043459");
			queries.add("C0000477_C0024408");
			queries.add("C1449659_C0024408");
			queries.add("C5139823_C0024408");
			queries.add("C0084707_C0024408");
			queries.add("C1569608_C0024408");
			queries.add("C0085217_C0024408");
			queries.add("C0700189_C0024408");
			queries.add("C0170531_C0024408");
			queries.add("C0281398_C0024408");
			queries.add("C0040815_C0024408");
			queries.add("C0034272_C1849508");
			queries.add("C0087162_C1849508");
			queries.add("C0000981_C0349653");
			queries.add("C0017237_C0349653");
			queries.add("C0022949_C0349653");
			queries.add("C0095278_C0349653");
			queries.add("C0048897_C0268465");
			queries.add("C0014806_C0027126");
			queries.add("C0600214_C0027126");
			queries.add("C0025598_C0027126");
			queries.add("C0016895_C0024408");
			queries.add("C0042291_C0024408");
			queries.add("C0016365_C0751882");
			
			
			HashMap<String,Float> methodWeight = new HashMap<String,Float>();
			methodWeight.put("AnyBURL", new Float(0.005));//0.005));
			//methodWeight.put("RGCN", new Float(0.0766));
			methodWeight.put("Paths", new Float(0.1));//0.1));
			methodWeight.put("BoP", new Float(0.10));//0.1));
			methodWeight.put("CompGCN", new Float(0.4));//0.4));
			
			String filePath1="/media/fot/USB STICK/FOT/Research/Graph analysis @dimokritos/GNN vs PRIMES for Link Prediction/";
			
			for (String query:queries) {
				HashMap<String,Float> drugWeight = new HashMap<String,Float> ();
				
				for (String method:methodWeight.keySet()) {
					FileReader filerdr1 = new FileReader(filePath1+"Ensemble of lists/"+method+"_top_500.csv");
					BufferedReader br1 = new BufferedReader(filerdr1);
	
					float w = methodWeight.get(method);
									
					String line = br1.readLine(); //labels
					while ((line = br1.readLine()) != null ) {
						String [] elems = line.split(",");
						String disease = elems[0];
						//System.out.println("method="+method+ " disease="+disease);
						String drug = elems[1];
						if (!((query.contains(disease)) && (query.contains(drug))))
							continue;
						
						for (int i=2; i<502; i++) {
							double orderWeight = 1.0-(i-2)*0.002;
							if (drugWeight.get(elems[i])==null)
								drugWeight.put(elems[i], new Float(w*orderWeight));
							else
								drugWeight.put(elems[i], new Float(w*orderWeight+drugWeight.get(elems[i])));
						}

					}
					br1.close();
					filerdr1.close();
				}
				//print all drug weights for the query
				System.out.println("Finished ensemble of query: "+ query);
				//just for debug
				if (query.equals("C0025887_C0027126"))
					for (String key: drugWeight.keySet()) 
						if ((key.equals("C0025887")) || (key.equals("C0209337")))
							System.out.println("drug: "+ key+ " score: "+drugWeight.get(key));
				//
				
				//save top-500 drug weights for the query
				FileWriter fw;
				fw = new FileWriter("/media/fot/USB STICK/FOT/Research/Graph analysis @dimokritos/GNN vs PRIMES for Link Prediction/Ensemble of lists/"+query+"-ensemble-top500.csv");
				fw.append("Rank,Drug_CUI,Score,Drug_name\n");
				int rank=1;
				while (rank<501) {
					//find drug with max score in each iteration
					String max_drug="";
					float max_score= 0;
					for (String drug: drugWeight.keySet()) {
						float score =drugWeight.get(drug);
						if (score>max_score) {
							max_score= score;
							max_drug= drug;
						}						
					}
					drugWeight.put(max_drug, new Float(0));
					fw.append(rank+","+max_drug+","+max_score+","+mapCUItoName(max_drug, ddiPath)+"\n");
					rank++;
				}	
				fw.close();
				
				
			}
			//calculate ensemble metrics
			mrr(filePath1);
			hitsAtN(5, filePath1);
			hitsAtN(10, filePath1);
			hitsAtN(100, filePath1);	

    }		

				
	static String mapCUItoName(String cui, String ddiPath) {
		
   		String name="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
					
			        String[] values = line.split("\t");
			        name = values[1];
			        break;
			        
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return name;
	}
	
	public static void mrr(String filePath1) throws Exception {
		FileReader groundfile = new FileReader(filePath1+"new-min-db-noDB/SIMPATHIC Drug-Disease testset.csv");
		BufferedReader br1 = new BufferedReader(groundfile);
		
		List<String> drug_disease = new LinkedList<String>();
		String line = br1.readLine(); //labels
		//Save all POS pairs in groundtruth
		while ((line = br1.readLine()).contains(",1") ) {
			String [] elems = line.split(",");
			String drug_dis = elems[0];
			drug_disease.add(drug_dis);			
		}
		br1.close();
		groundfile.close();
		
		double rr =0.0;
		//int j=1;
		for (String query:drug_disease) {
			
			String [] elems2 = query.split("_");
			String drug = elems2[0];
			String disease = elems2[1];
			//System.out.println("query "+ query+" no "+(j++));

			try {
				FileReader ensembleFile = new FileReader(filePath1+"Ensemble of lists/"+query+"-ensemble-top500.csv");
				BufferedReader ens = new BufferedReader(ensembleFile);
				int r=0;
				line = ens.readLine();//read labels line
				while ((line = ens.readLine())!=null ) {
					r++;
					if (line.contains(drug)) {
						//System.out.println("FOUND drug "+drug+" in disease "+disease+" list at rank: "+r);
						rr+=(1.0/(float)r);
						break;
					}
				}	
				ens.close();
				ensembleFile.close();
			} catch (Exception e) {System.out.println("Ranked drugs for disease not found");}
		}
		double mrr = rr/ ((float)drug_disease.size());
		System.out.println("MRR = divide "+rr+" / "+(float)drug_disease.size());
		System.out.println("ensemble MRR="+mrr);
		
	}
	public static void hitsAtN(int n, String filePath1) throws Exception {
		FileReader groundfile = new FileReader(filePath1+"new-min-db-noDB/SIMPATHIC Drug-Disease testset.csv");
		BufferedReader br1 = new BufferedReader(groundfile);
		
		List<String> drug_disease = new LinkedList<String>();
		String line = br1.readLine(); //labels
		//Save all POS pairs in groundtruth
		while ((line = br1.readLine()).contains(",1") ) {
			String [] elems = line.split(",");
			String drug_dis = elems[0];
			drug_disease.add(drug_dis);			
		}
		br1.close();
		groundfile.close();
		
		double hits =0.0;
		for (String query:drug_disease) {
			String [] elems2 = query.split("_");
			String drug = elems2[0];
			String disease = elems2[1];
			
			double h=0.0;
			try {
				FileReader ensembleFile = new FileReader(filePath1+"Ensemble of lists/"+query+"-ensemble-top500.csv");
				BufferedReader ens = new BufferedReader(ensembleFile);
				int r=0;
				line = ens.readLine();//read labels line
				while (((line = ens.readLine())!=null ) &&(r<n)) {
					r++;
					if (line.contains(drug)) {
						h=1.0;
						//System.out.println("HIT drug "+drug+" for disease "+disease+" list at rank: "+r);
						break;
					}
				}	
				ens.close();
				ensembleFile.close();
			} catch (Exception e) {System.out.println("Ranked drugs for disease not found");}	
			hits+=h;
		}
		double hitsAtN = hits/ (float)drug_disease.size();
		System.out.println("ensemble Hits@"+n+"="+hitsAtN);		
	}

}
