// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg meas[17];
ry(1.5765713111425983) q[0];
ry(1.582309018857911) q[1];
ry(1.5935292771638934) q[2];
ry(1.6140899847616947) q[3];
ry(1.6438454055481635) q[4];
ry(1.6611919395054304) q[5];
ry(1.6456790807969384) q[6];
ry(1.513257799767818) q[7];
cx q[7],q[6];
ry(0.8541072460194208) q[6];
cx q[7],q[6];
cx q[6],q[5];
ry(0.2608149580033374) q[5];
cx q[7],q[5];
ry(0.11717965633363425) q[5];
cx q[6],q[5];
ry(0.5301791166053285) q[5];
cx q[7],q[5];
cx q[5],q[4];
ry(0.08104360060191351) q[4];
cx q[6],q[4];
ry(0.020012001043265504) q[4];
cx q[5],q[4];
ry(0.1614793341268148) q[4];
cx q[7],q[4];
ry(0.08313433891710573) q[4];
cx q[5],q[4];
ry(0.008771669551962041) q[4];
cx q[6],q[4];
ry(0.040976012555095004) q[4];
cx q[5],q[4];
ry(0.3029932533202986) q[4];
cx q[7],q[4];
cx q[4],q[3];
ry(0.022654896250005163) q[3];
cx q[5],q[3];
ry(0.0035065493768621225) q[3];
cx q[4],q[3];
ry(0.045042141657073986) q[3];
cx q[6],q[3];
ry(0.013696914413324268) q[3];
cx q[4],q[3];
ry(0.0011182692042166487) q[3];
cx q[5],q[3];
ry(0.006878942014228506) q[3];
cx q[4],q[3];
ry(0.08782952500298713) q[3];
cx q[7],q[3];
ry(0.04773871371000189) q[3];
cx q[4],q[3];
ry(0.004088011156409013) q[3];
cx q[5],q[3];
ry(0.0006771520901947131) q[3];
cx q[4],q[3];
ry(0.008156349278967315) q[3];
cx q[6],q[3];
ry(0.024559869174431995) q[3];
cx q[4],q[3];
ry(0.002070238206548905) q[3];
cx q[5],q[3];
ry(0.012355959017209275) q[3];
cx q[4],q[3];
ry(0.15971324147727717) q[3];
cx q[7],q[3];
cx q[3],q[2];
ry(0.0058812799616279055) q[2];
cx q[4],q[2];
ry(0.0005033013085247173) q[2];
cx q[3],q[2];
ry(0.011736443963990428) q[2];
cx q[5],q[2];
ry(0.001982792102216774) q[2];
cx q[3],q[2];
ry(0.0001026993537026899) q[2];
cx q[4],q[2];
ry(0.0009944413476555558) q[2];
cx q[3],q[2];
ry(0.023268196170339234) q[2];
cx q[6],q[2];
ry(0.007482407830416857) q[2];
cx q[3],q[2];
ry(0.0003822634634095684) q[2];
cx q[4],q[2];
ry(4.5532993903044106e-05) q[2];
cx q[3],q[2];
ry(0.0007616770786908061) q[2];
cx q[5],q[2];
ry(0.0037865929359791745) q[2];
cx q[3],q[2];
ry(0.00019398613311762125) q[2];
cx q[4],q[2];
ry(0.001899077596448015) q[2];
cx q[3],q[2];
ry(0.04503586725250662) q[2];
cx q[7],q[2];
ry(0.024871747647917303) q[2];
cx q[3],q[2];
ry(0.0012025349853347356) q[2];
cx q[4],q[2];
ry(0.0001383328595894323) q[2];
cx q[3],q[2];
ry(0.002396213285530871) q[2];
cx q[5],q[2];
ry(0.0005420860049840243) q[2];
cx q[3],q[2];
ry(3.490754302189203e-05) q[2];
cx q[4],q[2];
ry(0.00027216459264109427) q[2];
cx q[3],q[2];
ry(0.004722820783543746) q[2];
cx q[6],q[2];
ry(0.012968905961824609) q[2];
cx q[3],q[2];
ry(0.0006356976881615878) q[2];
cx q[4],q[2];
ry(7.368688837608439e-05) q[2];
cx q[3],q[2];
ry(0.001266734539364045) q[2];
cx q[5],q[2];
ry(0.0065578691291765245) q[2];
cx q[3],q[2];
ry(0.000322505194865097) q[2];
cx q[4],q[2];
ry(0.003288314227176195) q[2];
cx q[3],q[2];
ry(0.0811305604860051) q[2];
cx q[7],q[2];
cx q[2],q[1];
ry(0.0014856810784273922) q[1];
cx q[3],q[1];
ry(6.555538256913795e-05) q[1];
cx q[2],q[1];
ry(0.0029695171860681935) q[1];
cx q[4],q[1];
ry(0.00026093054583226083) q[1];
cx q[2],q[1];
ry(7.325488031839633e-06) q[1];
cx q[3],q[1];
ry(0.00013059457851427592) q[1];
cx q[2],q[1];
ry(0.005924405143682365) q[1];
cx q[5],q[1];
ry(0.0010240159327806336) q[1];
cx q[2],q[1];
ry(2.8480147776052678e-05) q[1];
cx q[3],q[1];
ry(1.9869471996475374e-06) q[1];
cx q[2],q[1];
ry(5.687728086124777e-05) q[1];
cx q[4],q[1];
ry(0.0005139909814086269) q[1];
cx q[2],q[1];
ry(1.4322888106946263e-05) q[1];
cx q[3],q[1];
ry(0.00025724631796970876) q[1];
cx q[2],q[1];
ry(0.011735710313869482) q[1];
cx q[6],q[1];
ry(0.0038283366824547055) q[1];
cx q[2],q[1];
ry(0.00010318952507532009) q[1];
cx q[3],q[1];
ry(6.961245384642933e-06) q[1];
cx q[2],q[1];
ry(0.00020609714730335238) q[1];
cx q[4],q[1];
ry(2.758714805801976e-05) q[1];
cx q[2],q[1];
ry(1.1144869906942745e-06) q[1];
cx q[3],q[1];
ry(1.3819412787260954e-05) q[1];
cx q[2],q[1];
ry(0.0004099705676689769) q[1];
cx q[5],q[1];
ry(0.0019415520776169953) q[1];
cx q[2],q[1];
ry(5.269889313233578e-05) q[1];
cx q[3],q[1];
ry(3.5826033192715157e-06) q[1];
cx q[2],q[1];
ry(0.00010525164289981737) q[1];
cx q[4],q[1];
ry(0.0009743518384821709) q[1];
cx q[2],q[1];
ry(2.649524399353509e-05) q[1];
cx q[3],q[1];
ry(0.0004876279991696958) q[1];
cx q[2],q[1];
ry(0.0226683288874909) q[1];
cx q[7],q[1];
ry(0.012571300282149287) q[1];
cx q[2],q[1];
ry(0.0003135656583474235) q[1];
cx q[3],q[1];
ry(1.9518127345564268e-05) q[1];
cx q[2],q[1];
ry(0.0006263986132510231) q[1];
cx q[4],q[1];
ry(7.744339382689466e-05) q[1];
cx q[2],q[1];
ry(2.9002187561764925e-06) q[1];
cx q[3],q[1];
ry(3.8784736048968343e-05) q[1];
cx q[2],q[1];
ry(0.0012470090203194721) q[1];
cx q[5],q[1];
ry(0.0003002909957534085) q[1];
cx q[2],q[1];
ry(1.1142751813787016e-05) q[1];
cx q[3],q[1];
ry(9.582838538836624e-07) q[1];
cx q[2],q[1];
ry(2.2239151318260292e-05) q[1];
cx q[4],q[1];
ry(0.00015110136508981672) q[1];
cx q[2],q[1];
ry(5.617598716308597e-06) q[1];
cx q[3],q[1];
ry(7.567197922146371e-05) q[1];
cx q[2],q[1];
ry(0.002449904387736037) q[1];
cx q[6],q[1];
ry(0.006578726657583304) q[1];
cx q[2],q[1];
ry(0.00016757938526268737) q[1];
cx q[3],q[1];
ry(1.0681274804597712e-05) q[1];
cx q[2],q[1];
ry(0.00033474803339727036) q[1];
cx q[4],q[1];
ry(4.236490548128677e-05) q[1];
cx q[2],q[1];
ry(1.6251699034262879e-06) q[1];
cx q[3],q[1];
ry(2.1218546249878595e-05) q[1];
cx q[2],q[1];
ry(0.0006662528346959955) q[1];
cx q[5],q[1];
ry(0.0033314438914752893) q[1];
cx q[2],q[1];
ry(8.540083480304897e-05) q[1];
cx q[3],q[1];
ry(5.483175785820499e-06) q[1];
cx q[2],q[1];
ry(0.00017058926842028824) q[1];
cx q[4],q[1];
ry(0.0016711956386643821) q[1];
cx q[2],q[1];
ry(4.291234975715638e-05) q[1];
cx q[3],q[1];
ry(0.0008362891461063636) q[1];
cx q[2],q[1];
ry(0.040735197837101986) q[1];
cx q[7],q[1];
cx q[1],q[0];
ry(0.00037241299307855247) q[0];
cx q[2],q[0];
ry(8.283709165002318e-06) q[0];
cx q[1],q[0];
ry(0.0007447069517185212) q[0];
cx q[3],q[0];
ry(3.309098099368857e-05) q[0];
cx q[1],q[0];
ry(4.7513678380628477e-07) q[0];
cx q[2],q[0];
ry(1.6549878241456373e-05) q[0];
cx q[1],q[0];
ry(0.0014884640301892627) q[0];
cx q[4],q[0];
ry(0.00013167257674662203) q[0];
cx q[1],q[0];
ry(1.8848549069305776e-06) q[0];
cx q[2],q[0];
ry(6.931266519072388e-08) q[0];
cx q[1],q[0];
ry(3.7681350296414617e-06) q[0];
cx q[3],q[0];
ry(6.590555845842339e-05) q[0];
cx q[1],q[0];
ry(9.440009052785747e-07) q[0];
cx q[2],q[0];
ry(3.2961469930786746e-05) q[0];
cx q[1],q[0];
ry(0.0029694043250572955) q[0];
cx q[5],q[0];
ry(0.0005162265996584772) q[0];
cx q[1],q[0];
ry(7.306594838878133e-06) q[0];
cx q[2],q[0];
ry(2.649428137088683e-07) q[0];
cx q[1],q[0];
ry(1.4607266697504173e-05) q[0];
cx q[3],q[0];
ry(1.056639783870994e-06) q[0];
cx q[1],q[0];
ry(2.3595806718235135e-08) q[0];
cx q[2],q[0];
ry(5.286333052342829e-07) q[0];
cx q[1],q[0];
ry(2.9167380216583993e-05) q[0];
cx q[4],q[0];
ry(0.0002591674481557879) q[0];
cx q[1],q[0];
ry(3.676816859872478e-06) q[0];
cx q[2],q[0];
ry(1.337184558705684e-07) q[0];
cx q[1],q[0];
ry(7.350633995976963e-06) q[0];
cx q[3],q[0];
ry(0.0001297173625280021) q[0];
cx q[1],q[0];
ry(1.8414056848839455e-06) q[0];
cx q[2],q[0];
ry(6.487544608536205e-05) q[0];
cx q[1],q[0];
ry(0.005880843972388326) q[0];
cx q[6],q[0];
ry(0.0019253364994135033) q[0];
cx q[1],q[0];
ry(2.6296827382567278e-05) q[0];
cx q[2],q[0];
ry(9.141306227672397e-07) q[0];
cx q[1],q[0];
ry(5.2574145122968874e-05) q[0];
cx q[3],q[0];
ry(3.6466885777325375e-06) q[0];
cx q[1],q[0];
ry(7.775090245243366e-08) q[0];
cx q[2],q[0];
ry(1.8243284409594218e-06) q[0];
cx q[1],q[0];
ry(0.00010499290347979612) q[0];
cx q[4],q[0];
ry(1.4432994050142534e-05) q[0];
cx q[1],q[0];
ry(3.0653154911575164e-07) q[0];
cx q[2],q[0];
ry(1.5437643892390884e-08) q[0];
cx q[1],q[0];
ry(6.12613646137905e-07) q[0];
cx q[3],q[0];
ry(7.231919901047601e-06) q[0];
cx q[1],q[0];
ry(1.5371468781860687e-07) q[0];
cx q[2],q[0];
ry(3.617898888502724e-06) q[0];
cx q[1],q[0];
ry(0.00020876414962642253) q[0];
cx q[5],q[0];
ry(0.0009769805966447637) q[0];
cx q[1],q[0];
ry(1.3451453174259664e-05) q[0];
cx q[2],q[0];
ry(4.722748541199917e-07) q[0];
cx q[1],q[0];
ry(2.6892710349953053e-05) q[0];
cx q[3],q[0];
ry(1.883895619135545e-06) q[0];
cx q[1],q[0];
ry(4.062974507529349e-08) q[0];
cx q[2],q[0];
ry(9.424686025910622e-07) q[0];
cx q[1],q[0];
ry(5.370422289405208e-05) q[0];
cx q[4],q[0];
ry(0.0004903700522213908) q[0];
cx q[1],q[0];
ry(6.7662337234000725e-06) q[0];
cx q[2],q[0];
ry(2.3821033867665875e-07) q[0];
cx q[1],q[0];
ry(1.3527308067223798e-05) q[0];
cx q[3],q[0];
ry(0.000245423103683598) q[0];
cx q[1],q[0];
ry(3.3882722861311443e-06) q[0];
cx q[2],q[0];
ry(0.0001227414111295358) q[0];
cx q[1],q[0];
ry(0.01135331865199559) q[0];
cx q[7],q[0];
ry(0.006302929143255951) q[0];
cx q[1],q[0];
ry(7.923038682092903e-05) q[0];
cx q[2],q[0];
ry(2.515453778044463e-06) q[0];
cx q[1],q[0];
ry(0.0001584118097954397) q[0];
cx q[3],q[0];
ry(1.0039216665005263e-05) q[0];
cx q[1],q[0];
ry(1.9524683191746162e-07) q[0];
cx q[2],q[0];
ry(5.0218696704754096e-06) q[0];
cx q[1],q[0];
ry(0.0003164333692978298) q[0];
cx q[4],q[0];
ry(3.980235198225435e-05) q[0];
cx q[1],q[0];
ry(7.714914165117548e-07) q[0];
cx q[2],q[0];
ry(3.557292083469776e-08) q[0];
cx q[1],q[0];
ry(1.5420290970166728e-06) q[0];
cx q[3],q[0];
ry(1.9936719825426608e-05) q[0];
cx q[1],q[0];
ry(3.8669845045177786e-07) q[0];
cx q[2],q[0];
ry(9.972824711618894e-06) q[0];
cx q[1],q[0];
ry(0.0006297902629391339) q[0];
cx q[5],q[0];
ry(0.00015393870038225843) q[0];
cx q[1],q[0];
ry(2.9474412455744725e-06) q[0];
cx q[2],q[0];
ry(1.3401904052148805e-07) q[0];
cx q[1],q[0];
ry(5.891344129232978e-06) q[0];
cx q[3],q[0];
ry(5.339652222394187e-07) q[0];
cx q[1],q[0];
ry(1.4082592373482195e-08) q[0];
cx q[2],q[0];
ry(2.67193914649938e-07) q[0];
cx q[1],q[0];
ry(1.1754551406971403e-05) q[0];
cx q[4],q[0];
ry(7.750163745491936e-05) q[0];
cx q[1],q[0];
ry(1.4877469644569707e-06) q[0];
cx q[2],q[0];
ry(6.784953827249085e-08) q[0];
cx q[1],q[0];
ry(2.973696580779034e-06) q[0];
cx q[3],q[0];
ry(3.881861413957896e-05) q[0];
cx q[1],q[0];
ry(7.456690044455039e-07) q[0];
cx q[2],q[0];
ry(1.9417822099486448e-05) q[0];
cx q[1],q[0];
ry(0.001236291272995866) q[0];
cx q[6],q[0];
ry(0.0033014326275069214) q[0];
cx q[1],q[0];
ry(4.2459866659873974e-05) q[0];
cx q[2],q[0];
ry(1.385707865942526e-06) q[0];
cx q[1],q[0];
ry(8.489191154083253e-05) q[0];
cx q[3],q[0];
ry(5.52956377024294e-06) q[0];
cx q[1],q[0];
ry(1.1091758930423246e-07) q[0];
cx q[2],q[0];
ry(2.7661095917015655e-06) q[0];
cx q[1],q[0];
ry(0.0001695621356805413) q[0];
cx q[4],q[0];
ry(2.1910395025203505e-05) q[0];
cx q[1],q[0];
ry(4.3791368400242225e-07) q[0];
cx q[2],q[0];
ry(2.086208698154346e-08) q[0];
cx q[1],q[0];
ry(8.752490220607922e-07) q[0];
cx q[3],q[0];
ry(1.0976041372515871e-05) q[0];
cx q[1],q[0];
ry(2.1953454552164908e-07) q[0];
cx q[2],q[0];
ry(5.490639842030409e-06) q[0];
cx q[1],q[0];
ry(0.0003373783694843209) q[0];
cx q[5],q[0];
ry(0.0016724633419179617) q[0];
cx q[1],q[0];
ry(2.1663343933206458e-05) q[0];
cx q[2],q[0];
ry(7.134338909156601e-07) q[0];
cx q[1],q[0];
ry(4.331220851522613e-05) q[0];
cx q[3],q[0];
ry(2.84674706476154e-06) q[0];
cx q[1],q[0];
ry(5.772020346406226e-08) q[0];
cx q[2],q[0];
ry(1.424072889173733e-06) q[0];
cx q[1],q[0];
ry(8.650905548563889e-05) q[0];
cx q[4],q[0];
ry(0.0008390728533807859) q[0];
cx q[1],q[0];
ry(1.0889235346703968e-05) q[0];
cx q[2],q[0];
ry(3.595017839565229e-07) q[0];
cx q[1],q[0];
ry(2.177115251606604e-05) q[0];
cx q[3],q[0];
ry(0.00041989575048531624) q[0];
cx q[1],q[0];
ry(5.451930803233168e-06) q[0];
cx q[2],q[0];
ry(0.0002099929242292166) q[0];
cx q[1],q[0];
ry(0.0203892068744099) q[0];
cx q[7],q[0];
ry(3*pi/8) q[8];
cry(0) q[0],q[8];
cry(0) q[1],q[8];
x q[1];
cry(0) q[2],q[8];
cry(0) q[3],q[8];
cry(0) q[4],q[8];
x q[4];
cry(0) q[5],q[8];
cry(0) q[6],q[8];
cry(0) q[7],q[8];
x q[7];
x q[9];
x q[10];
x q[11];
ccx q[1],q[10],q[11];
ccx q[2],q[11],q[12];
ccx q[3],q[12],q[13];
x q[13];
x q[14];
ccx q[4],q[13],q[14];
ccx q[5],q[14],q[15];
ccx q[6],q[15],q[16];
x q[16];
ccx q[7],q[16],q[9];
x q[16];
ccx q[6],q[15],q[16];
ccx q[5],q[14],q[15];
x q[14];
ccx q[4],q[13],q[14];
x q[13];
ccx q[3],q[12],q[13];
ccx q[2],q[11],q[12];
x q[11];
ccx q[1],q[10],q[11];
x q[1];
x q[4];
x q[7];
cx q[9],q[8];
u(0.2942523647695424,0,0) q[8];
cx q[9],q[8];
u3(0.2942523647695424,-pi,-pi) q[8];
cx q[9],q[8];
u(-0.0013469636205260128,0,0) q[8];
cx q[9],q[8];
u(0.0013469636205260128,0,0) q[8];
ccx q[9],q[0],q[8];
cx q[9],q[8];
u(0.0013469636205260128,0,0) q[8];
cx q[9],q[8];
u(-0.0013469636205260128,0,0) q[8];
ccx q[9],q[0],q[8];
cx q[9],q[8];
u(-0.0026939272410520256,0,0) q[8];
cx q[9],q[8];
u(0.0026939272410520256,0,0) q[8];
ccx q[9],q[1],q[8];
cx q[9],q[8];
u(0.0026939272410520256,0,0) q[8];
cx q[9],q[8];
u(-0.0026939272410520256,0,0) q[8];
ccx q[9],q[1],q[8];
x q[1];
ccx q[1],q[10],q[11];
x q[11];
cx q[9],q[8];
u(-0.005387854482104051,0,0) q[8];
cx q[9],q[8];
u(0.005387854482104051,0,0) q[8];
ccx q[9],q[2],q[8];
cx q[9],q[8];
u(0.005387854482104051,0,0) q[8];
cx q[9],q[8];
u(-0.005387854482104051,0,0) q[8];
ccx q[9],q[2],q[8];
ccx q[2],q[11],q[12];
cx q[9],q[8];
u(-0.010775708964208102,0,0) q[8];
cx q[9],q[8];
u(0.010775708964208102,0,0) q[8];
ccx q[9],q[3],q[8];
cx q[9],q[8];
u(0.010775708964208102,0,0) q[8];
cx q[9],q[8];
u(-0.010775708964208102,0,0) q[8];
ccx q[9],q[3],q[8];
ccx q[3],q[12],q[13];
x q[13];
cx q[9],q[8];
u(-0.021551417928416205,0,0) q[8];
cx q[9],q[8];
u(0.021551417928416205,0,0) q[8];
ccx q[9],q[4],q[8];
cx q[9],q[8];
u(0.021551417928416205,0,0) q[8];
cx q[9],q[8];
u(-0.021551417928416205,0,0) q[8];
ccx q[9],q[4],q[8];
x q[4];
ccx q[4],q[13],q[14];
x q[14];
cx q[9],q[8];
u(-0.04310283585683241,0,0) q[8];
cx q[9],q[8];
u(0.04310283585683241,0,0) q[8];
ccx q[9],q[5],q[8];
cx q[9],q[8];
u(0.04310283585683241,0,0) q[8];
cx q[9],q[8];
u(-0.04310283585683241,0,0) q[8];
ccx q[9],q[5],q[8];
ccx q[5],q[14],q[15];
cx q[9],q[8];
u(-0.08620567171366482,0,0) q[8];
cx q[9],q[8];
u(0.08620567171366482,0,0) q[8];
ccx q[9],q[6],q[8];
cx q[9],q[8];
u(0.08620567171366482,0,0) q[8];
cx q[9],q[8];
u(-0.08620567171366482,0,0) q[8];
ccx q[9],q[6],q[8];
ccx q[6],q[15],q[16];
x q[16];
cx q[9],q[8];
u(-0.17241134342732964,0,0) q[8];
cx q[9],q[8];
u(0.17241134342732964,0,0) q[8];
ccx q[9],q[7],q[8];
cx q[9],q[8];
u(0.17241134342732964,0,0) q[8];
cx q[9],q[8];
u(-0.17241134342732964,0,0) q[8];
ccx q[9],q[7],q[8];
x q[7];
ccx q[7],q[16],q[9];
x q[16];
ccx q[6],q[15],q[16];
ccx q[5],q[14],q[15];
ccx q[4],q[13],q[14];
x q[13];
x q[14];
ccx q[3],q[12],q[13];
ccx q[2],q[11],q[12];
ccx q[1],q[10],q[11];
x q[1];
x q[10];
x q[11];
x q[4];
x q[7];
x q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
