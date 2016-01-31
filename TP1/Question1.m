B = 1; F = 2; G = 3; D = 4; FT = 5;

names = cell(1,5);
names{B}  = 'Battery';
names{F}  = 'Fuel';
names{G}  = 'Gauge';
names{D}  = 'Distance';
names{FT} = 'BatteryFillTank';

dgm = zeros(5,5);
dgm(B,G) = 1;
dgm(F,G) = 1;
dgm(G,[D, FT]) = 1;

CPDs{B} = tabularCpdCreate(reshape([0.1 0.9], 2, 1));
CPDs{F} = tabularCpdCreate(reshape([0.1 0.9], 2, 1));
CPDs{G} = tabularCpdCreate(reshape([0.9 0.8 0.8 0.2 0.1 0.2 0.2 0.8], 2, 2, 2));
CPDs{D} = tabularCpdCreate(reshape([0.95 0.7 0.05 0.3], 2, 2));
CPDs{FT} = tabularCpdCreate(reshape([0.2 0.6 0.8 0.4], 2, 2));

dgm = dgmCreate(dgm, CPDs, 'nodenames', names, 'infEngine', 'jtree');
joint = dgmInferQuery(dgm, [B,F,G,D,FT]);


fprintf('------------Explaining Away------------\n');
fprintf('En fixant B et en regardant le résultat sur\n');
fprintf('la probabilité de F on voit que rien ne change.\n');

clamped = sparsevec(B,1,5);
FGivenB = tabularFactorCondition(joint, F, clamped);
fprintf('p(F|B=0)=%f\n', FGivenB.T(1));

clamped = sparsevec(B,2,5);
FGivenB = tabularFactorCondition(joint, F, clamped);
fprintf('p(F|B=1)=%f\n', FGivenB.T(1));
fprintf('\n');

fprintf('Par contre, si on fixe G, B modifie maintenant F\n');
fprintf('On explique le résultat parce que B explain away G pour F.\n');
clamped = sparsevec([B G],2,5);
FGivenBG = tabularFactorCondition(joint, F, clamped);
fprintf('p(F|B=1,G=1)=%f\n', FGivenBG.T(1));
clamped = sparsevec([B G],[1, 2],5);
FGivenBG = tabularFactorCondition(joint, F, clamped);
fprintf('p(F|B=0,G=1)=%f\n', FGivenBG.T(1));

fprintf('\n');
fprintf('------------Serial Blocking------------\n');

fprintf('On voit que la modification de B a une influence sur D.\n');

clamped = sparsevec(B, 1, 5);
DGivenB = tabularFactorCondition(joint, D, clamped);
fprintf('p(D|B=0)=%f\n', DGivenB.T(1));
clamped = sparsevec(B, 2, 5);
DGivenB = tabularFactorCondition(joint, D, clamped);
fprintf('p(D|B=1)=%f\n', DGivenB.T(1));
fprintf('\n');

fprintf('En fixant G, on bloque linfluence de B sur D\n');
clamped =sparsevec([B G], [2 1], 5);
DGivenBG = tabularFactorCondition(joint, D, clamped);
fprintf('p(D|B=1, G=0)=%f\n', DGivenBG.T(1));
clamped =sparsevec([B G], [1 1], 5);
DGivenBG = tabularFactorCondition(joint, D, clamped);
fprintf('p(D|B=0, G=0)=%f\n', DGivenBG.T(1));

fprintf('------------Divergent Blocking------------\n');
fprintf('D a une influence sur FT.\n');
clamped = sparsevec(D,1,5);
FTGivenD = tabularFactorCondition(joint, FT, clamped);
fprintf('p(FT|D=0)=%f\n',FTGivenD.T(1));
clamped = sparsevec(D,2,5);
FTGivenD = tabularFactorCondition(joint, FT, clamped);
fprintf('p(FT|D=1)=%f\n',FTGivenD.T(1));
fprintf('\n');

fprintf('En fixant G, on bloque linfluence de D sur FT\n');
clamped = sparsevec([D G],1,5);
FTGivenDG = tabularFactorCondition(joint, FT, clamped);
fprintf('p(FT|D=0,G=0)=%f\n',FTGivenDG.T(1));
clamped = sparsevec([D G],[2 1],5);
FTGivenDG = tabularFactorCondition(joint, FT, clamped);
fprintf('p(FT|D=1,G=0)=%f\n',FTGivenDG.T(1));

