C = 1; T = 2; A = 3; M = 4; J = 5;

names = cell(1,5);
names{C} = 'Cambriolage';
names{T} = 'Tremblement';
names{A} = 'Alarme';
names{M} = 'MarieAppelle';
names{J} = 'JeanAppelle';

dgm = zeros(5,5);
dgm(C, A) = 1;
dgm(T, A) = 1;
dgm(T, J) = 1;
dgm(A, M) = 1;
dgm(A, J) = 1;

CPDs{C} = tabularCpdCreate(reshape([0.999 0.001], 2, 1));
CPDs{T} = tabularCpdCreate(reshape([0.998 0.002], 2, 1));
CPDs{A} = tabularCpdCreate(reshape([0.999, 0.06, 0.71, 0.05, 0.001, 0.94, 0.29, 0.95], 2, 2, 2));
CPDs{M} = tabularCpdCreate(reshape([0.95 0.1 0.05 0.9], 2, 2));
CPDs{J} = tabularCpdCreate(reshape([0.999 0.999 0.5 0.95 0.001 0.001 0.5 0.05], 2, 2, 2));

dgm = dgmCreate(dgm, CPDs, 'nodenames', names, 'infEngine', 'jtree');

joint = dgmInferQuery(dgm, [C,T,A,M,J]);
fprintf('2.b)\n');
v1 = [1, 2];
v2 = [1, 2];
v3 = [1, 2];
v4 = [1, 2];
v5 = [1, 2];
samples = combvec(v1, v2, v3, v4, v5);
jointProbability = zeros(1, size(samples, 2));

for col = 1:size(samples, 2)
	sample = samples(:, col);
	clamped = zeros(1,5);
	pC = tabularFactorCondition(joint, C, clamped);

	clamped = zeros(1,5);
	pT = tabularFactorCondition(joint, T, clamped);

	clamped = zeros(1,5);
	clamped(C) = sample(C);
	clamped(T) = sample(T);
	pAGivenCT = tabularFactorCondition(joint, A, clamped);

	clamped = zeros(1,5);
	clamped(A) = sample(A);
	pMGivenA = tabularFactorCondition(joint, M, clamped);

	clamped = zeros(1,5);
	clamped(A) = sample(A);
	clamped(T) = sample(T);
	pJGivenAT = tabularFactorCondition(joint, J, clamped);

	jointProbability(:, col) = pC.T(sample(C)) * pT.T(sample(T)) * pAGivenCT.T(sample(A)) * pMGivenA.T(sample(M)) * pJGivenAT.T(sample(J));
end

bar(jointProbability)
fprintf('2.c)\n');

joint = dgmInferQuery(dgm, [C,T,A,M,J]);
clamped = sparsevec([M J],[2 1],5);
CGivenMJ = tabularFactorCondition(joint, C, clamped);
fprintf('p(C=1|M=1,J=0)=%f\n',CGivenMJ.T(2));

clamped = sparsevec([M J],[1 2],5);
CGivenMJ = tabularFactorCondition(joint, C, clamped);
fprintf('p(C=1|M=0,J=1)=%f\n',CGivenMJ.T(2));

clamped = sparsevec([M J],[2 2],5);
CGivenMJ = tabularFactorCondition(joint, C, clamped);
fprintf('p(C=1|M=1,J=1)=%f\n',CGivenMJ.T(2));

clamped = sparsevec([M J],[1 1],5);
CGivenMJ = tabularFactorCondition(joint, C, clamped);
fprintf('p(C=1|M=0,J=0)=%f\n',CGivenMJ.T(2));

clamped = sparsevec(M,2,5);
CGivenM = tabularFactorCondition(joint, C, clamped);
fprintf('p(C=1|M=1)=%f\n',CGivenM.T(2));

clamped = sparsevec(J,2,5);
CGivenJ = tabularFactorCondition(joint, C, clamped);
fprintf('p(C=1|J=1)=%f\n',CGivenJ.T(2));

fprintf('2.d)\n');
clamped = zeros(5,1);
pC = tabularFactorCondition(joint, C, clamped);
fprintf('p(C=1)=%f\n',pC.T(2));

clamped = zeros(5,1);
pT = tabularFactorCondition(joint, T, clamped);
fprintf('p(T=1)=%f\n',pT.T(2));

clamped = zeros(5,1);
pA = tabularFactorCondition(joint, A, clamped);
fprintf('p(A=1)=%f\n',pA.T(2));

clamped = zeros(5,1);
pM = tabularFactorCondition(joint, M, clamped);
fprintf('p(M=1)=%f\n',pM.T(2));

clamped = zeros(5,1);
pJ = tabularFactorCondition(joint, J, clamped);
fprintf('p(J=1)=%f\n',pJ.T(2));
