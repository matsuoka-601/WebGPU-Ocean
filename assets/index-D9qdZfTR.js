var ln=Object.defineProperty;var un=(f,h,v)=>h in f?ln(f,h,{enumerable:!0,configurable:!0,writable:!0,value:v}):f[h]=v;var b=(f,h,v)=>un(f,typeof h!="symbol"?h+"":h,v);(function(){const h=document.createElement("link").relList;if(h&&h.supports&&h.supports("modulepreload"))return;for(const M of document.querySelectorAll('link[rel="modulepreload"]'))z(M);new MutationObserver(M=>{for(const G of M)if(G.type==="childList")for(const I of G.addedNodes)I.tagName==="LINK"&&I.rel==="modulepreload"&&z(I)}).observe(document,{childList:!0,subtree:!0});function v(M){const G={};return M.integrity&&(G.integrity=M.integrity),M.referrerPolicy&&(G.referrerPolicy=M.referrerPolicy),M.crossOrigin==="use-credentials"?G.credentials="include":M.crossOrigin==="anonymous"?G.credentials="omit":G.credentials="same-origin",G}function z(M){if(M.ep)return;M.ep=!0;const G=v(M);fetch(M.href,G)}})();function fn(f,h){return class extends f{constructor(...v){super(...v),h(this)}}}const dn=fn(Array,f=>f.fill(0));let A=1e-6;function pn(f){function h(e=0,o=0){const n=new f(2);return e!==void 0&&(n[0]=e,o!==void 0&&(n[1]=o)),n}const v=h;function z(e,o,n){const i=n??new f(2);return i[0]=e,i[1]=o,i}function M(e,o){const n=o??new f(2);return n[0]=Math.ceil(e[0]),n[1]=Math.ceil(e[1]),n}function G(e,o){const n=o??new f(2);return n[0]=Math.floor(e[0]),n[1]=Math.floor(e[1]),n}function I(e,o){const n=o??new f(2);return n[0]=Math.round(e[0]),n[1]=Math.round(e[1]),n}function E(e,o=0,n=1,i){const a=i??new f(2);return a[0]=Math.min(n,Math.max(o,e[0])),a[1]=Math.min(n,Math.max(o,e[1])),a}function F(e,o,n){const i=n??new f(2);return i[0]=e[0]+o[0],i[1]=e[1]+o[1],i}function K(e,o,n,i){const a=i??new f(2);return a[0]=e[0]+o[0]*n,a[1]=e[1]+o[1]*n,a}function Z(e,o){const n=e[0],i=e[1],a=o[0],g=o[1],y=Math.sqrt(n*n+i*i),c=Math.sqrt(a*a+g*g),l=y*c,x=l&&de(e,o)/l;return Math.acos(x)}function U(e,o,n){const i=n??new f(2);return i[0]=e[0]-o[0],i[1]=e[1]-o[1],i}const N=U;function q(e,o){return Math.abs(e[0]-o[0])<A&&Math.abs(e[1]-o[1])<A}function Q(e,o){return e[0]===o[0]&&e[1]===o[1]}function X(e,o,n,i){const a=i??new f(2);return a[0]=e[0]+n*(o[0]-e[0]),a[1]=e[1]+n*(o[1]-e[1]),a}function Y(e,o,n,i){const a=i??new f(2);return a[0]=e[0]+n[0]*(o[0]-e[0]),a[1]=e[1]+n[1]*(o[1]-e[1]),a}function ee(e,o,n){const i=n??new f(2);return i[0]=Math.max(e[0],o[0]),i[1]=Math.max(e[1],o[1]),i}function se(e,o,n){const i=n??new f(2);return i[0]=Math.min(e[0],o[0]),i[1]=Math.min(e[1],o[1]),i}function ie(e,o,n){const i=n??new f(2);return i[0]=e[0]*o,i[1]=e[1]*o,i}const le=ie;function te(e,o,n){const i=n??new f(2);return i[0]=e[0]/o,i[1]=e[1]/o,i}function oe(e,o){const n=o??new f(2);return n[0]=1/e[0],n[1]=1/e[1],n}const ue=oe;function fe(e,o,n){const i=n??new f(3),a=e[0]*o[1]-e[1]*o[0];return i[0]=0,i[1]=0,i[2]=a,i}function de(e,o){return e[0]*o[0]+e[1]*o[1]}function ne(e){const o=e[0],n=e[1];return Math.sqrt(o*o+n*n)}const xe=ne;function k(e){const o=e[0],n=e[1];return o*o+n*n}const V=k;function T(e,o){const n=e[0]-o[0],i=e[1]-o[1];return Math.sqrt(n*n+i*i)}const J=T;function W(e,o){const n=e[0]-o[0],i=e[1]-o[1];return n*n+i*i}const $=W;function ce(e,o){const n=o??new f(2),i=e[0],a=e[1],g=Math.sqrt(i*i+a*a);return g>1e-5?(n[0]=i/g,n[1]=a/g):(n[0]=0,n[1]=0),n}function ge(e,o){const n=o??new f(2);return n[0]=-e[0],n[1]=-e[1],n}function j(e,o){const n=o??new f(2);return n[0]=e[0],n[1]=e[1],n}const ae=j;function _e(e,o,n){const i=n??new f(2);return i[0]=e[0]*o[0],i[1]=e[1]*o[1],i}const be=_e;function ve(e,o,n){const i=n??new f(2);return i[0]=e[0]/o[0],i[1]=e[1]/o[1],i}const Te=ve;function C(e=1,o){const n=o??new f(2),i=Math.random()*2*Math.PI;return n[0]=Math.cos(i)*e,n[1]=Math.sin(i)*e,n}function r(e){const o=e??new f(2);return o[0]=0,o[1]=0,o}function p(e,o,n){const i=n??new f(2),a=e[0],g=e[1];return i[0]=a*o[0]+g*o[4]+o[12],i[1]=a*o[1]+g*o[5]+o[13],i}function t(e,o,n){const i=n??new f(2),a=e[0],g=e[1];return i[0]=o[0]*a+o[4]*g+o[8],i[1]=o[1]*a+o[5]*g+o[9],i}function s(e,o,n,i){const a=i??new f(2),g=e[0]-o[0],y=e[1]-o[1],c=Math.sin(n),l=Math.cos(n);return a[0]=g*l-y*c+o[0],a[1]=g*c+y*l+o[1],a}function u(e,o,n){const i=n??new f(2);return ce(e,i),ie(i,o,i)}function d(e,o,n){const i=n??new f(2);return ne(e)>o?u(e,o,i):j(e,i)}function w(e,o,n){const i=n??new f(2);return X(e,o,.5,i)}return{create:h,fromValues:v,set:z,ceil:M,floor:G,round:I,clamp:E,add:F,addScaled:K,angle:Z,subtract:U,sub:N,equalsApproximately:q,equals:Q,lerp:X,lerpV:Y,max:ee,min:se,mulScalar:ie,scale:le,divScalar:te,inverse:oe,invert:ue,cross:fe,dot:de,length:ne,len:xe,lengthSq:k,lenSq:V,distance:T,dist:J,distanceSq:W,distSq:$,normalize:ce,negate:ge,copy:j,clone:ae,multiply:_e,mul:be,divide:ve,div:Te,random:C,zero:r,transformMat4:p,transformMat3:t,rotate:s,setLength:u,truncate:d,midpoint:w}}const en=new Map;function an(f){let h=en.get(f);return h||(h=pn(f),en.set(f,h)),h}function hn(f){function h(c,l,x){const _=new f(3);return c!==void 0&&(_[0]=c,l!==void 0&&(_[1]=l,x!==void 0&&(_[2]=x))),_}const v=h;function z(c,l,x,_){const m=_??new f(3);return m[0]=c,m[1]=l,m[2]=x,m}function M(c,l){const x=l??new f(3);return x[0]=Math.ceil(c[0]),x[1]=Math.ceil(c[1]),x[2]=Math.ceil(c[2]),x}function G(c,l){const x=l??new f(3);return x[0]=Math.floor(c[0]),x[1]=Math.floor(c[1]),x[2]=Math.floor(c[2]),x}function I(c,l){const x=l??new f(3);return x[0]=Math.round(c[0]),x[1]=Math.round(c[1]),x[2]=Math.round(c[2]),x}function E(c,l=0,x=1,_){const m=_??new f(3);return m[0]=Math.min(x,Math.max(l,c[0])),m[1]=Math.min(x,Math.max(l,c[1])),m[2]=Math.min(x,Math.max(l,c[2])),m}function F(c,l,x){const _=x??new f(3);return _[0]=c[0]+l[0],_[1]=c[1]+l[1],_[2]=c[2]+l[2],_}function K(c,l,x,_){const m=_??new f(3);return m[0]=c[0]+l[0]*x,m[1]=c[1]+l[1]*x,m[2]=c[2]+l[2]*x,m}function Z(c,l){const x=c[0],_=c[1],m=c[2],P=l[0],D=l[1],S=l[2],O=Math.sqrt(x*x+_*_+m*m),B=Math.sqrt(P*P+D*D+S*S),R=O*B,L=R&&de(c,l)/R;return Math.acos(L)}function U(c,l,x){const _=x??new f(3);return _[0]=c[0]-l[0],_[1]=c[1]-l[1],_[2]=c[2]-l[2],_}const N=U;function q(c,l){return Math.abs(c[0]-l[0])<A&&Math.abs(c[1]-l[1])<A&&Math.abs(c[2]-l[2])<A}function Q(c,l){return c[0]===l[0]&&c[1]===l[1]&&c[2]===l[2]}function X(c,l,x,_){const m=_??new f(3);return m[0]=c[0]+x*(l[0]-c[0]),m[1]=c[1]+x*(l[1]-c[1]),m[2]=c[2]+x*(l[2]-c[2]),m}function Y(c,l,x,_){const m=_??new f(3);return m[0]=c[0]+x[0]*(l[0]-c[0]),m[1]=c[1]+x[1]*(l[1]-c[1]),m[2]=c[2]+x[2]*(l[2]-c[2]),m}function ee(c,l,x){const _=x??new f(3);return _[0]=Math.max(c[0],l[0]),_[1]=Math.max(c[1],l[1]),_[2]=Math.max(c[2],l[2]),_}function se(c,l,x){const _=x??new f(3);return _[0]=Math.min(c[0],l[0]),_[1]=Math.min(c[1],l[1]),_[2]=Math.min(c[2],l[2]),_}function ie(c,l,x){const _=x??new f(3);return _[0]=c[0]*l,_[1]=c[1]*l,_[2]=c[2]*l,_}const le=ie;function te(c,l,x){const _=x??new f(3);return _[0]=c[0]/l,_[1]=c[1]/l,_[2]=c[2]/l,_}function oe(c,l){const x=l??new f(3);return x[0]=1/c[0],x[1]=1/c[1],x[2]=1/c[2],x}const ue=oe;function fe(c,l,x){const _=x??new f(3),m=c[2]*l[0]-c[0]*l[2],P=c[0]*l[1]-c[1]*l[0];return _[0]=c[1]*l[2]-c[2]*l[1],_[1]=m,_[2]=P,_}function de(c,l){return c[0]*l[0]+c[1]*l[1]+c[2]*l[2]}function ne(c){const l=c[0],x=c[1],_=c[2];return Math.sqrt(l*l+x*x+_*_)}const xe=ne;function k(c){const l=c[0],x=c[1],_=c[2];return l*l+x*x+_*_}const V=k;function T(c,l){const x=c[0]-l[0],_=c[1]-l[1],m=c[2]-l[2];return Math.sqrt(x*x+_*_+m*m)}const J=T;function W(c,l){const x=c[0]-l[0],_=c[1]-l[1],m=c[2]-l[2];return x*x+_*_+m*m}const $=W;function ce(c,l){const x=l??new f(3),_=c[0],m=c[1],P=c[2],D=Math.sqrt(_*_+m*m+P*P);return D>1e-5?(x[0]=_/D,x[1]=m/D,x[2]=P/D):(x[0]=0,x[1]=0,x[2]=0),x}function ge(c,l){const x=l??new f(3);return x[0]=-c[0],x[1]=-c[1],x[2]=-c[2],x}function j(c,l){const x=l??new f(3);return x[0]=c[0],x[1]=c[1],x[2]=c[2],x}const ae=j;function _e(c,l,x){const _=x??new f(3);return _[0]=c[0]*l[0],_[1]=c[1]*l[1],_[2]=c[2]*l[2],_}const be=_e;function ve(c,l,x){const _=x??new f(3);return _[0]=c[0]/l[0],_[1]=c[1]/l[1],_[2]=c[2]/l[2],_}const Te=ve;function C(c=1,l){const x=l??new f(3),_=Math.random()*2*Math.PI,m=Math.random()*2-1,P=Math.sqrt(1-m*m)*c;return x[0]=Math.cos(_)*P,x[1]=Math.sin(_)*P,x[2]=m*c,x}function r(c){const l=c??new f(3);return l[0]=0,l[1]=0,l[2]=0,l}function p(c,l,x){const _=x??new f(3),m=c[0],P=c[1],D=c[2],S=l[3]*m+l[7]*P+l[11]*D+l[15]||1;return _[0]=(l[0]*m+l[4]*P+l[8]*D+l[12])/S,_[1]=(l[1]*m+l[5]*P+l[9]*D+l[13])/S,_[2]=(l[2]*m+l[6]*P+l[10]*D+l[14])/S,_}function t(c,l,x){const _=x??new f(3),m=c[0],P=c[1],D=c[2];return _[0]=m*l[0*4+0]+P*l[1*4+0]+D*l[2*4+0],_[1]=m*l[0*4+1]+P*l[1*4+1]+D*l[2*4+1],_[2]=m*l[0*4+2]+P*l[1*4+2]+D*l[2*4+2],_}function s(c,l,x){const _=x??new f(3),m=c[0],P=c[1],D=c[2];return _[0]=m*l[0]+P*l[4]+D*l[8],_[1]=m*l[1]+P*l[5]+D*l[9],_[2]=m*l[2]+P*l[6]+D*l[10],_}function u(c,l,x){const _=x??new f(3),m=l[0],P=l[1],D=l[2],S=l[3]*2,O=c[0],B=c[1],R=c[2],L=P*R-D*B,H=D*O-m*R,re=m*B-P*O;return _[0]=O+L*S+(P*re-D*H)*2,_[1]=B+H*S+(D*L-m*re)*2,_[2]=R+re*S+(m*H-P*L)*2,_}function d(c,l){const x=l??new f(3);return x[0]=c[12],x[1]=c[13],x[2]=c[14],x}function w(c,l,x){const _=x??new f(3),m=l*4;return _[0]=c[m+0],_[1]=c[m+1],_[2]=c[m+2],_}function e(c,l){const x=l??new f(3),_=c[0],m=c[1],P=c[2],D=c[4],S=c[5],O=c[6],B=c[8],R=c[9],L=c[10];return x[0]=Math.sqrt(_*_+m*m+P*P),x[1]=Math.sqrt(D*D+S*S+O*O),x[2]=Math.sqrt(B*B+R*R+L*L),x}function o(c,l,x,_){const m=_??new f(3),P=[],D=[];return P[0]=c[0]-l[0],P[1]=c[1]-l[1],P[2]=c[2]-l[2],D[0]=P[0],D[1]=P[1]*Math.cos(x)-P[2]*Math.sin(x),D[2]=P[1]*Math.sin(x)+P[2]*Math.cos(x),m[0]=D[0]+l[0],m[1]=D[1]+l[1],m[2]=D[2]+l[2],m}function n(c,l,x,_){const m=_??new f(3),P=[],D=[];return P[0]=c[0]-l[0],P[1]=c[1]-l[1],P[2]=c[2]-l[2],D[0]=P[2]*Math.sin(x)+P[0]*Math.cos(x),D[1]=P[1],D[2]=P[2]*Math.cos(x)-P[0]*Math.sin(x),m[0]=D[0]+l[0],m[1]=D[1]+l[1],m[2]=D[2]+l[2],m}function i(c,l,x,_){const m=_??new f(3),P=[],D=[];return P[0]=c[0]-l[0],P[1]=c[1]-l[1],P[2]=c[2]-l[2],D[0]=P[0]*Math.cos(x)-P[1]*Math.sin(x),D[1]=P[0]*Math.sin(x)+P[1]*Math.cos(x),D[2]=P[2],m[0]=D[0]+l[0],m[1]=D[1]+l[1],m[2]=D[2]+l[2],m}function a(c,l,x){const _=x??new f(3);return ce(c,_),ie(_,l,_)}function g(c,l,x){const _=x??new f(3);return ne(c)>l?a(c,l,_):j(c,_)}function y(c,l,x){const _=x??new f(3);return X(c,l,.5,_)}return{create:h,fromValues:v,set:z,ceil:M,floor:G,round:I,clamp:E,add:F,addScaled:K,angle:Z,subtract:U,sub:N,equalsApproximately:q,equals:Q,lerp:X,lerpV:Y,max:ee,min:se,mulScalar:ie,scale:le,divScalar:te,inverse:oe,invert:ue,cross:fe,dot:de,length:ne,len:xe,lengthSq:k,lenSq:V,distance:T,dist:J,distanceSq:W,distSq:$,normalize:ce,negate:ge,copy:j,clone:ae,multiply:_e,mul:be,divide:ve,div:Te,random:C,zero:r,transformMat4:p,transformMat4Upper3x3:t,transformMat3:s,transformQuat:u,getTranslation:d,getAxis:w,getScaling:e,rotateX:o,rotateY:n,rotateZ:i,setLength:a,truncate:g,midpoint:y}}const nn=new Map;function Xe(f){let h=nn.get(f);return h||(h=hn(f),nn.set(f,h)),h}function xn(f){const h=an(f),v=Xe(f);function z(r,p,t,s,u,d,w,e,o){const n=new f(12);return n[3]=0,n[7]=0,n[11]=0,r!==void 0&&(n[0]=r,p!==void 0&&(n[1]=p,t!==void 0&&(n[2]=t,s!==void 0&&(n[4]=s,u!==void 0&&(n[5]=u,d!==void 0&&(n[6]=d,w!==void 0&&(n[8]=w,e!==void 0&&(n[9]=e,o!==void 0&&(n[10]=o))))))))),n}function M(r,p,t,s,u,d,w,e,o,n){const i=n??new f(12);return i[0]=r,i[1]=p,i[2]=t,i[3]=0,i[4]=s,i[5]=u,i[6]=d,i[7]=0,i[8]=w,i[9]=e,i[10]=o,i[11]=0,i}function G(r,p){const t=p??new f(12);return t[0]=r[0],t[1]=r[1],t[2]=r[2],t[3]=0,t[4]=r[4],t[5]=r[5],t[6]=r[6],t[7]=0,t[8]=r[8],t[9]=r[9],t[10]=r[10],t[11]=0,t}function I(r,p){const t=p??new f(12),s=r[0],u=r[1],d=r[2],w=r[3],e=s+s,o=u+u,n=d+d,i=s*e,a=u*e,g=u*o,y=d*e,c=d*o,l=d*n,x=w*e,_=w*o,m=w*n;return t[0]=1-g-l,t[1]=a+m,t[2]=y-_,t[3]=0,t[4]=a-m,t[5]=1-i-l,t[6]=c+x,t[7]=0,t[8]=y+_,t[9]=c-x,t[10]=1-i-g,t[11]=0,t}function E(r,p){const t=p??new f(12);return t[0]=-r[0],t[1]=-r[1],t[2]=-r[2],t[4]=-r[4],t[5]=-r[5],t[6]=-r[6],t[8]=-r[8],t[9]=-r[9],t[10]=-r[10],t}function F(r,p){const t=p??new f(12);return t[0]=r[0],t[1]=r[1],t[2]=r[2],t[4]=r[4],t[5]=r[5],t[6]=r[6],t[8]=r[8],t[9]=r[9],t[10]=r[10],t}const K=F;function Z(r,p){return Math.abs(r[0]-p[0])<A&&Math.abs(r[1]-p[1])<A&&Math.abs(r[2]-p[2])<A&&Math.abs(r[4]-p[4])<A&&Math.abs(r[5]-p[5])<A&&Math.abs(r[6]-p[6])<A&&Math.abs(r[8]-p[8])<A&&Math.abs(r[9]-p[9])<A&&Math.abs(r[10]-p[10])<A}function U(r,p){return r[0]===p[0]&&r[1]===p[1]&&r[2]===p[2]&&r[4]===p[4]&&r[5]===p[5]&&r[6]===p[6]&&r[8]===p[8]&&r[9]===p[9]&&r[10]===p[10]}function N(r){const p=r??new f(12);return p[0]=1,p[1]=0,p[2]=0,p[4]=0,p[5]=1,p[6]=0,p[8]=0,p[9]=0,p[10]=1,p}function q(r,p){const t=p??new f(12);if(t===r){let g;return g=r[1],r[1]=r[4],r[4]=g,g=r[2],r[2]=r[8],r[8]=g,g=r[6],r[6]=r[9],r[9]=g,t}const s=r[0*4+0],u=r[0*4+1],d=r[0*4+2],w=r[1*4+0],e=r[1*4+1],o=r[1*4+2],n=r[2*4+0],i=r[2*4+1],a=r[2*4+2];return t[0]=s,t[1]=w,t[2]=n,t[4]=u,t[5]=e,t[6]=i,t[8]=d,t[9]=o,t[10]=a,t}function Q(r,p){const t=p??new f(12),s=r[0*4+0],u=r[0*4+1],d=r[0*4+2],w=r[1*4+0],e=r[1*4+1],o=r[1*4+2],n=r[2*4+0],i=r[2*4+1],a=r[2*4+2],g=a*e-o*i,y=-a*w+o*n,c=i*w-e*n,l=1/(s*g+u*y+d*c);return t[0]=g*l,t[1]=(-a*u+d*i)*l,t[2]=(o*u-d*e)*l,t[4]=y*l,t[5]=(a*s-d*n)*l,t[6]=(-o*s+d*w)*l,t[8]=c*l,t[9]=(-i*s+u*n)*l,t[10]=(e*s-u*w)*l,t}function X(r){const p=r[0],t=r[0*4+1],s=r[0*4+2],u=r[1*4+0],d=r[1*4+1],w=r[1*4+2],e=r[2*4+0],o=r[2*4+1],n=r[2*4+2];return p*(d*n-o*w)-u*(t*n-o*s)+e*(t*w-d*s)}const Y=Q;function ee(r,p,t){const s=t??new f(12),u=r[0],d=r[1],w=r[2],e=r[4],o=r[5],n=r[6],i=r[8],a=r[9],g=r[10],y=p[0],c=p[1],l=p[2],x=p[4],_=p[5],m=p[6],P=p[8],D=p[9],S=p[10];return s[0]=u*y+e*c+i*l,s[1]=d*y+o*c+a*l,s[2]=w*y+n*c+g*l,s[4]=u*x+e*_+i*m,s[5]=d*x+o*_+a*m,s[6]=w*x+n*_+g*m,s[8]=u*P+e*D+i*S,s[9]=d*P+o*D+a*S,s[10]=w*P+n*D+g*S,s}const se=ee;function ie(r,p,t){const s=t??N();return r!==s&&(s[0]=r[0],s[1]=r[1],s[2]=r[2],s[4]=r[4],s[5]=r[5],s[6]=r[6]),s[8]=p[0],s[9]=p[1],s[10]=1,s}function le(r,p){const t=p??h.create();return t[0]=r[8],t[1]=r[9],t}function te(r,p,t){const s=t??h.create(),u=p*4;return s[0]=r[u+0],s[1]=r[u+1],s}function oe(r,p,t,s){const u=s===r?r:F(r,s),d=t*4;return u[d+0]=p[0],u[d+1]=p[1],u}function ue(r,p){const t=p??h.create(),s=r[0],u=r[1],d=r[4],w=r[5];return t[0]=Math.sqrt(s*s+u*u),t[1]=Math.sqrt(d*d+w*w),t}function fe(r,p){const t=p??v.create(),s=r[0],u=r[1],d=r[2],w=r[4],e=r[5],o=r[6],n=r[8],i=r[9],a=r[10];return t[0]=Math.sqrt(s*s+u*u+d*d),t[1]=Math.sqrt(w*w+e*e+o*o),t[2]=Math.sqrt(n*n+i*i+a*a),t}function de(r,p){const t=p??new f(12);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=1,t[6]=0,t[8]=r[0],t[9]=r[1],t[10]=1,t}function ne(r,p,t){const s=t??new f(12),u=p[0],d=p[1],w=r[0],e=r[1],o=r[2],n=r[1*4+0],i=r[1*4+1],a=r[1*4+2],g=r[2*4+0],y=r[2*4+1],c=r[2*4+2];return r!==s&&(s[0]=w,s[1]=e,s[2]=o,s[4]=n,s[5]=i,s[6]=a),s[8]=w*u+n*d+g,s[9]=e*u+i*d+y,s[10]=o*u+a*d+c,s}function xe(r,p){const t=p??new f(12),s=Math.cos(r),u=Math.sin(r);return t[0]=s,t[1]=u,t[2]=0,t[4]=-u,t[5]=s,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function k(r,p,t){const s=t??new f(12),u=r[0*4+0],d=r[0*4+1],w=r[0*4+2],e=r[1*4+0],o=r[1*4+1],n=r[1*4+2],i=Math.cos(p),a=Math.sin(p);return s[0]=i*u+a*e,s[1]=i*d+a*o,s[2]=i*w+a*n,s[4]=i*e-a*u,s[5]=i*o-a*d,s[6]=i*n-a*w,r!==s&&(s[8]=r[8],s[9]=r[9],s[10]=r[10]),s}function V(r,p){const t=p??new f(12),s=Math.cos(r),u=Math.sin(r);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=s,t[6]=u,t[8]=0,t[9]=-u,t[10]=s,t}function T(r,p,t){const s=t??new f(12),u=r[4],d=r[5],w=r[6],e=r[8],o=r[9],n=r[10],i=Math.cos(p),a=Math.sin(p);return s[4]=i*u+a*e,s[5]=i*d+a*o,s[6]=i*w+a*n,s[8]=i*e-a*u,s[9]=i*o-a*d,s[10]=i*n-a*w,r!==s&&(s[0]=r[0],s[1]=r[1],s[2]=r[2]),s}function J(r,p){const t=p??new f(12),s=Math.cos(r),u=Math.sin(r);return t[0]=s,t[1]=0,t[2]=-u,t[4]=0,t[5]=1,t[6]=0,t[8]=u,t[9]=0,t[10]=s,t}function W(r,p,t){const s=t??new f(12),u=r[0*4+0],d=r[0*4+1],w=r[0*4+2],e=r[2*4+0],o=r[2*4+1],n=r[2*4+2],i=Math.cos(p),a=Math.sin(p);return s[0]=i*u-a*e,s[1]=i*d-a*o,s[2]=i*w-a*n,s[8]=i*e+a*u,s[9]=i*o+a*d,s[10]=i*n+a*w,r!==s&&(s[4]=r[4],s[5]=r[5],s[6]=r[6]),s}const $=xe,ce=k;function ge(r,p){const t=p??new f(12);return t[0]=r[0],t[1]=0,t[2]=0,t[4]=0,t[5]=r[1],t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function j(r,p,t){const s=t??new f(12),u=p[0],d=p[1];return s[0]=u*r[0*4+0],s[1]=u*r[0*4+1],s[2]=u*r[0*4+2],s[4]=d*r[1*4+0],s[5]=d*r[1*4+1],s[6]=d*r[1*4+2],r!==s&&(s[8]=r[8],s[9]=r[9],s[10]=r[10]),s}function ae(r,p){const t=p??new f(12);return t[0]=r[0],t[1]=0,t[2]=0,t[4]=0,t[5]=r[1],t[6]=0,t[8]=0,t[9]=0,t[10]=r[2],t}function _e(r,p,t){const s=t??new f(12),u=p[0],d=p[1],w=p[2];return s[0]=u*r[0*4+0],s[1]=u*r[0*4+1],s[2]=u*r[0*4+2],s[4]=d*r[1*4+0],s[5]=d*r[1*4+1],s[6]=d*r[1*4+2],s[8]=w*r[2*4+0],s[9]=w*r[2*4+1],s[10]=w*r[2*4+2],s}function be(r,p){const t=p??new f(12);return t[0]=r,t[1]=0,t[2]=0,t[4]=0,t[5]=r,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function ve(r,p,t){const s=t??new f(12);return s[0]=p*r[0*4+0],s[1]=p*r[0*4+1],s[2]=p*r[0*4+2],s[4]=p*r[1*4+0],s[5]=p*r[1*4+1],s[6]=p*r[1*4+2],r!==s&&(s[8]=r[8],s[9]=r[9],s[10]=r[10]),s}function Te(r,p){const t=p??new f(12);return t[0]=r,t[1]=0,t[2]=0,t[4]=0,t[5]=r,t[6]=0,t[8]=0,t[9]=0,t[10]=r,t}function C(r,p,t){const s=t??new f(12);return s[0]=p*r[0*4+0],s[1]=p*r[0*4+1],s[2]=p*r[0*4+2],s[4]=p*r[1*4+0],s[5]=p*r[1*4+1],s[6]=p*r[1*4+2],s[8]=p*r[2*4+0],s[9]=p*r[2*4+1],s[10]=p*r[2*4+2],s}return{clone:K,create:z,set:M,fromMat4:G,fromQuat:I,negate:E,copy:F,equalsApproximately:Z,equals:U,identity:N,transpose:q,inverse:Q,invert:Y,determinant:X,mul:se,multiply:ee,setTranslation:ie,getTranslation:le,getAxis:te,setAxis:oe,getScaling:ue,get3DScaling:fe,translation:de,translate:ne,rotation:xe,rotate:k,rotationX:V,rotateX:T,rotationY:J,rotateY:W,rotationZ:$,rotateZ:ce,scaling:ge,scale:j,uniformScaling:be,uniformScale:ve,scaling3D:ae,scale3D:_e,uniformScaling3D:Te,uniformScale3D:C}}const tn=new Map;function gn(f){let h=tn.get(f);return h||(h=xn(f),tn.set(f,h)),h}function _n(f){const h=Xe(f);function v(e,o,n,i,a,g,y,c,l,x,_,m,P,D,S,O){const B=new f(16);return e!==void 0&&(B[0]=e,o!==void 0&&(B[1]=o,n!==void 0&&(B[2]=n,i!==void 0&&(B[3]=i,a!==void 0&&(B[4]=a,g!==void 0&&(B[5]=g,y!==void 0&&(B[6]=y,c!==void 0&&(B[7]=c,l!==void 0&&(B[8]=l,x!==void 0&&(B[9]=x,_!==void 0&&(B[10]=_,m!==void 0&&(B[11]=m,P!==void 0&&(B[12]=P,D!==void 0&&(B[13]=D,S!==void 0&&(B[14]=S,O!==void 0&&(B[15]=O)))))))))))))))),B}function z(e,o,n,i,a,g,y,c,l,x,_,m,P,D,S,O,B){const R=B??new f(16);return R[0]=e,R[1]=o,R[2]=n,R[3]=i,R[4]=a,R[5]=g,R[6]=y,R[7]=c,R[8]=l,R[9]=x,R[10]=_,R[11]=m,R[12]=P,R[13]=D,R[14]=S,R[15]=O,R}function M(e,o){const n=o??new f(16);return n[0]=e[0],n[1]=e[1],n[2]=e[2],n[3]=0,n[4]=e[4],n[5]=e[5],n[6]=e[6],n[7]=0,n[8]=e[8],n[9]=e[9],n[10]=e[10],n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,n}function G(e,o){const n=o??new f(16),i=e[0],a=e[1],g=e[2],y=e[3],c=i+i,l=a+a,x=g+g,_=i*c,m=a*c,P=a*l,D=g*c,S=g*l,O=g*x,B=y*c,R=y*l,L=y*x;return n[0]=1-P-O,n[1]=m+L,n[2]=D-R,n[3]=0,n[4]=m-L,n[5]=1-_-O,n[6]=S+B,n[7]=0,n[8]=D+R,n[9]=S-B,n[10]=1-_-P,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,n}function I(e,o){const n=o??new f(16);return n[0]=-e[0],n[1]=-e[1],n[2]=-e[2],n[3]=-e[3],n[4]=-e[4],n[5]=-e[5],n[6]=-e[6],n[7]=-e[7],n[8]=-e[8],n[9]=-e[9],n[10]=-e[10],n[11]=-e[11],n[12]=-e[12],n[13]=-e[13],n[14]=-e[14],n[15]=-e[15],n}function E(e,o){const n=o??new f(16);return n[0]=e[0],n[1]=e[1],n[2]=e[2],n[3]=e[3],n[4]=e[4],n[5]=e[5],n[6]=e[6],n[7]=e[7],n[8]=e[8],n[9]=e[9],n[10]=e[10],n[11]=e[11],n[12]=e[12],n[13]=e[13],n[14]=e[14],n[15]=e[15],n}const F=E;function K(e,o){return Math.abs(e[0]-o[0])<A&&Math.abs(e[1]-o[1])<A&&Math.abs(e[2]-o[2])<A&&Math.abs(e[3]-o[3])<A&&Math.abs(e[4]-o[4])<A&&Math.abs(e[5]-o[5])<A&&Math.abs(e[6]-o[6])<A&&Math.abs(e[7]-o[7])<A&&Math.abs(e[8]-o[8])<A&&Math.abs(e[9]-o[9])<A&&Math.abs(e[10]-o[10])<A&&Math.abs(e[11]-o[11])<A&&Math.abs(e[12]-o[12])<A&&Math.abs(e[13]-o[13])<A&&Math.abs(e[14]-o[14])<A&&Math.abs(e[15]-o[15])<A}function Z(e,o){return e[0]===o[0]&&e[1]===o[1]&&e[2]===o[2]&&e[3]===o[3]&&e[4]===o[4]&&e[5]===o[5]&&e[6]===o[6]&&e[7]===o[7]&&e[8]===o[8]&&e[9]===o[9]&&e[10]===o[10]&&e[11]===o[11]&&e[12]===o[12]&&e[13]===o[13]&&e[14]===o[14]&&e[15]===o[15]}function U(e){const o=e??new f(16);return o[0]=1,o[1]=0,o[2]=0,o[3]=0,o[4]=0,o[5]=1,o[6]=0,o[7]=0,o[8]=0,o[9]=0,o[10]=1,o[11]=0,o[12]=0,o[13]=0,o[14]=0,o[15]=1,o}function N(e,o){const n=o??new f(16);if(n===e){let H;return H=e[1],e[1]=e[4],e[4]=H,H=e[2],e[2]=e[8],e[8]=H,H=e[3],e[3]=e[12],e[12]=H,H=e[6],e[6]=e[9],e[9]=H,H=e[7],e[7]=e[13],e[13]=H,H=e[11],e[11]=e[14],e[14]=H,n}const i=e[0*4+0],a=e[0*4+1],g=e[0*4+2],y=e[0*4+3],c=e[1*4+0],l=e[1*4+1],x=e[1*4+2],_=e[1*4+3],m=e[2*4+0],P=e[2*4+1],D=e[2*4+2],S=e[2*4+3],O=e[3*4+0],B=e[3*4+1],R=e[3*4+2],L=e[3*4+3];return n[0]=i,n[1]=c,n[2]=m,n[3]=O,n[4]=a,n[5]=l,n[6]=P,n[7]=B,n[8]=g,n[9]=x,n[10]=D,n[11]=R,n[12]=y,n[13]=_,n[14]=S,n[15]=L,n}function q(e,o){const n=o??new f(16),i=e[0*4+0],a=e[0*4+1],g=e[0*4+2],y=e[0*4+3],c=e[1*4+0],l=e[1*4+1],x=e[1*4+2],_=e[1*4+3],m=e[2*4+0],P=e[2*4+1],D=e[2*4+2],S=e[2*4+3],O=e[3*4+0],B=e[3*4+1],R=e[3*4+2],L=e[3*4+3],H=D*L,re=R*S,pe=x*L,he=R*_,we=x*S,me=D*_,ye=g*L,Pe=R*y,Me=g*S,De=D*y,Ge=g*_,Be=x*y,Se=m*B,Re=O*P,Ie=c*B,Ee=O*l,Oe=c*P,Le=m*l,Ve=i*B,Ne=O*a,We=i*P,qe=m*a,He=i*l,Ke=c*a,$e=H*l+he*P+we*B-(re*l+pe*P+me*B),Qe=re*a+ye*P+De*B-(H*a+Pe*P+Me*B),Ce=pe*a+Pe*l+Ge*B-(he*a+ye*l+Be*B),Je=me*a+Me*l+Be*P-(we*a+De*l+Ge*P),ze=1/(i*$e+c*Qe+m*Ce+O*Je);return n[0]=ze*$e,n[1]=ze*Qe,n[2]=ze*Ce,n[3]=ze*Je,n[4]=ze*(re*c+pe*m+me*O-(H*c+he*m+we*O)),n[5]=ze*(H*i+Pe*m+Me*O-(re*i+ye*m+De*O)),n[6]=ze*(he*i+ye*c+Be*O-(pe*i+Pe*c+Ge*O)),n[7]=ze*(we*i+De*c+Ge*m-(me*i+Me*c+Be*m)),n[8]=ze*(Se*_+Ee*S+Oe*L-(Re*_+Ie*S+Le*L)),n[9]=ze*(Re*y+Ve*S+qe*L-(Se*y+Ne*S+We*L)),n[10]=ze*(Ie*y+Ne*_+He*L-(Ee*y+Ve*_+Ke*L)),n[11]=ze*(Le*y+We*_+Ke*S-(Oe*y+qe*_+He*S)),n[12]=ze*(Ie*D+Le*R+Re*x-(Oe*R+Se*x+Ee*D)),n[13]=ze*(We*R+Se*g+Ne*D-(Ve*D+qe*R+Re*g)),n[14]=ze*(Ve*x+Ke*R+Ee*g-(He*R+Ie*g+Ne*x)),n[15]=ze*(He*D+Oe*g+qe*x-(We*x+Ke*D+Le*g)),n}function Q(e){const o=e[0],n=e[0*4+1],i=e[0*4+2],a=e[0*4+3],g=e[1*4+0],y=e[1*4+1],c=e[1*4+2],l=e[1*4+3],x=e[2*4+0],_=e[2*4+1],m=e[2*4+2],P=e[2*4+3],D=e[3*4+0],S=e[3*4+1],O=e[3*4+2],B=e[3*4+3],R=m*B,L=O*P,H=c*B,re=O*l,pe=c*P,he=m*l,we=i*B,me=O*a,ye=i*P,Pe=m*a,Me=i*l,De=c*a,Ge=R*y+re*_+pe*S-(L*y+H*_+he*S),Be=L*n+we*_+Pe*S-(R*n+me*_+ye*S),Se=H*n+me*y+Me*S-(re*n+we*y+De*S),Re=he*n+ye*y+De*_-(pe*n+Pe*y+Me*_);return o*Ge+g*Be+x*Se+D*Re}const X=q;function Y(e,o,n){const i=n??new f(16),a=e[0],g=e[1],y=e[2],c=e[3],l=e[4],x=e[5],_=e[6],m=e[7],P=e[8],D=e[9],S=e[10],O=e[11],B=e[12],R=e[13],L=e[14],H=e[15],re=o[0],pe=o[1],he=o[2],we=o[3],me=o[4],ye=o[5],Pe=o[6],Me=o[7],De=o[8],Ge=o[9],Be=o[10],Se=o[11],Re=o[12],Ie=o[13],Ee=o[14],Oe=o[15];return i[0]=a*re+l*pe+P*he+B*we,i[1]=g*re+x*pe+D*he+R*we,i[2]=y*re+_*pe+S*he+L*we,i[3]=c*re+m*pe+O*he+H*we,i[4]=a*me+l*ye+P*Pe+B*Me,i[5]=g*me+x*ye+D*Pe+R*Me,i[6]=y*me+_*ye+S*Pe+L*Me,i[7]=c*me+m*ye+O*Pe+H*Me,i[8]=a*De+l*Ge+P*Be+B*Se,i[9]=g*De+x*Ge+D*Be+R*Se,i[10]=y*De+_*Ge+S*Be+L*Se,i[11]=c*De+m*Ge+O*Be+H*Se,i[12]=a*Re+l*Ie+P*Ee+B*Oe,i[13]=g*Re+x*Ie+D*Ee+R*Oe,i[14]=y*Re+_*Ie+S*Ee+L*Oe,i[15]=c*Re+m*Ie+O*Ee+H*Oe,i}const ee=Y;function se(e,o,n){const i=n??U();return e!==i&&(i[0]=e[0],i[1]=e[1],i[2]=e[2],i[3]=e[3],i[4]=e[4],i[5]=e[5],i[6]=e[6],i[7]=e[7],i[8]=e[8],i[9]=e[9],i[10]=e[10],i[11]=e[11]),i[12]=o[0],i[13]=o[1],i[14]=o[2],i[15]=1,i}function ie(e,o){const n=o??h.create();return n[0]=e[12],n[1]=e[13],n[2]=e[14],n}function le(e,o,n){const i=n??h.create(),a=o*4;return i[0]=e[a+0],i[1]=e[a+1],i[2]=e[a+2],i}function te(e,o,n,i){const a=i===e?i:E(e,i),g=n*4;return a[g+0]=o[0],a[g+1]=o[1],a[g+2]=o[2],a}function oe(e,o){const n=o??h.create(),i=e[0],a=e[1],g=e[2],y=e[4],c=e[5],l=e[6],x=e[8],_=e[9],m=e[10];return n[0]=Math.sqrt(i*i+a*a+g*g),n[1]=Math.sqrt(y*y+c*c+l*l),n[2]=Math.sqrt(x*x+_*_+m*m),n}function ue(e,o,n,i,a){const g=a??new f(16),y=Math.tan(Math.PI*.5-.5*e);if(g[0]=y/o,g[1]=0,g[2]=0,g[3]=0,g[4]=0,g[5]=y,g[6]=0,g[7]=0,g[8]=0,g[9]=0,g[11]=-1,g[12]=0,g[13]=0,g[15]=0,Number.isFinite(i)){const c=1/(n-i);g[10]=i*c,g[14]=i*n*c}else g[10]=-1,g[14]=-n;return g}function fe(e,o,n,i=1/0,a){const g=a??new f(16),y=1/Math.tan(e*.5);if(g[0]=y/o,g[1]=0,g[2]=0,g[3]=0,g[4]=0,g[5]=y,g[6]=0,g[7]=0,g[8]=0,g[9]=0,g[11]=-1,g[12]=0,g[13]=0,g[15]=0,i===1/0)g[10]=0,g[14]=n;else{const c=1/(i-n);g[10]=n*c,g[14]=i*n*c}return g}function de(e,o,n,i,a,g,y){const c=y??new f(16);return c[0]=2/(o-e),c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2/(i-n),c[6]=0,c[7]=0,c[8]=0,c[9]=0,c[10]=1/(a-g),c[11]=0,c[12]=(o+e)/(e-o),c[13]=(i+n)/(n-i),c[14]=a/(a-g),c[15]=1,c}function ne(e,o,n,i,a,g,y){const c=y??new f(16),l=o-e,x=i-n,_=a-g;return c[0]=2*a/l,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/x,c[6]=0,c[7]=0,c[8]=(e+o)/l,c[9]=(i+n)/x,c[10]=g/_,c[11]=-1,c[12]=0,c[13]=0,c[14]=a*g/_,c[15]=0,c}function xe(e,o,n,i,a,g=1/0,y){const c=y??new f(16),l=o-e,x=i-n;if(c[0]=2*a/l,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/x,c[6]=0,c[7]=0,c[8]=(e+o)/l,c[9]=(i+n)/x,c[11]=-1,c[12]=0,c[13]=0,c[15]=0,g===1/0)c[10]=0,c[14]=a;else{const _=1/(g-a);c[10]=a*_,c[14]=g*a*_}return c}const k=h.create(),V=h.create(),T=h.create();function J(e,o,n,i){const a=i??new f(16);return h.normalize(h.subtract(o,e,T),T),h.normalize(h.cross(n,T,k),k),h.normalize(h.cross(T,k,V),V),a[0]=k[0],a[1]=k[1],a[2]=k[2],a[3]=0,a[4]=V[0],a[5]=V[1],a[6]=V[2],a[7]=0,a[8]=T[0],a[9]=T[1],a[10]=T[2],a[11]=0,a[12]=e[0],a[13]=e[1],a[14]=e[2],a[15]=1,a}function W(e,o,n,i){const a=i??new f(16);return h.normalize(h.subtract(e,o,T),T),h.normalize(h.cross(n,T,k),k),h.normalize(h.cross(T,k,V),V),a[0]=k[0],a[1]=k[1],a[2]=k[2],a[3]=0,a[4]=V[0],a[5]=V[1],a[6]=V[2],a[7]=0,a[8]=T[0],a[9]=T[1],a[10]=T[2],a[11]=0,a[12]=e[0],a[13]=e[1],a[14]=e[2],a[15]=1,a}function $(e,o,n,i){const a=i??new f(16);return h.normalize(h.subtract(e,o,T),T),h.normalize(h.cross(n,T,k),k),h.normalize(h.cross(T,k,V),V),a[0]=k[0],a[1]=V[0],a[2]=T[0],a[3]=0,a[4]=k[1],a[5]=V[1],a[6]=T[1],a[7]=0,a[8]=k[2],a[9]=V[2],a[10]=T[2],a[11]=0,a[12]=-(k[0]*e[0]+k[1]*e[1]+k[2]*e[2]),a[13]=-(V[0]*e[0]+V[1]*e[1]+V[2]*e[2]),a[14]=-(T[0]*e[0]+T[1]*e[1]+T[2]*e[2]),a[15]=1,a}function ce(e,o){const n=o??new f(16);return n[0]=1,n[1]=0,n[2]=0,n[3]=0,n[4]=0,n[5]=1,n[6]=0,n[7]=0,n[8]=0,n[9]=0,n[10]=1,n[11]=0,n[12]=e[0],n[13]=e[1],n[14]=e[2],n[15]=1,n}function ge(e,o,n){const i=n??new f(16),a=o[0],g=o[1],y=o[2],c=e[0],l=e[1],x=e[2],_=e[3],m=e[1*4+0],P=e[1*4+1],D=e[1*4+2],S=e[1*4+3],O=e[2*4+0],B=e[2*4+1],R=e[2*4+2],L=e[2*4+3],H=e[3*4+0],re=e[3*4+1],pe=e[3*4+2],he=e[3*4+3];return e!==i&&(i[0]=c,i[1]=l,i[2]=x,i[3]=_,i[4]=m,i[5]=P,i[6]=D,i[7]=S,i[8]=O,i[9]=B,i[10]=R,i[11]=L),i[12]=c*a+m*g+O*y+H,i[13]=l*a+P*g+B*y+re,i[14]=x*a+D*g+R*y+pe,i[15]=_*a+S*g+L*y+he,i}function j(e,o){const n=o??new f(16),i=Math.cos(e),a=Math.sin(e);return n[0]=1,n[1]=0,n[2]=0,n[3]=0,n[4]=0,n[5]=i,n[6]=a,n[7]=0,n[8]=0,n[9]=-a,n[10]=i,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,n}function ae(e,o,n){const i=n??new f(16),a=e[4],g=e[5],y=e[6],c=e[7],l=e[8],x=e[9],_=e[10],m=e[11],P=Math.cos(o),D=Math.sin(o);return i[4]=P*a+D*l,i[5]=P*g+D*x,i[6]=P*y+D*_,i[7]=P*c+D*m,i[8]=P*l-D*a,i[9]=P*x-D*g,i[10]=P*_-D*y,i[11]=P*m-D*c,e!==i&&(i[0]=e[0],i[1]=e[1],i[2]=e[2],i[3]=e[3],i[12]=e[12],i[13]=e[13],i[14]=e[14],i[15]=e[15]),i}function _e(e,o){const n=o??new f(16),i=Math.cos(e),a=Math.sin(e);return n[0]=i,n[1]=0,n[2]=-a,n[3]=0,n[4]=0,n[5]=1,n[6]=0,n[7]=0,n[8]=a,n[9]=0,n[10]=i,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,n}function be(e,o,n){const i=n??new f(16),a=e[0*4+0],g=e[0*4+1],y=e[0*4+2],c=e[0*4+3],l=e[2*4+0],x=e[2*4+1],_=e[2*4+2],m=e[2*4+3],P=Math.cos(o),D=Math.sin(o);return i[0]=P*a-D*l,i[1]=P*g-D*x,i[2]=P*y-D*_,i[3]=P*c-D*m,i[8]=P*l+D*a,i[9]=P*x+D*g,i[10]=P*_+D*y,i[11]=P*m+D*c,e!==i&&(i[4]=e[4],i[5]=e[5],i[6]=e[6],i[7]=e[7],i[12]=e[12],i[13]=e[13],i[14]=e[14],i[15]=e[15]),i}function ve(e,o){const n=o??new f(16),i=Math.cos(e),a=Math.sin(e);return n[0]=i,n[1]=a,n[2]=0,n[3]=0,n[4]=-a,n[5]=i,n[6]=0,n[7]=0,n[8]=0,n[9]=0,n[10]=1,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,n}function Te(e,o,n){const i=n??new f(16),a=e[0*4+0],g=e[0*4+1],y=e[0*4+2],c=e[0*4+3],l=e[1*4+0],x=e[1*4+1],_=e[1*4+2],m=e[1*4+3],P=Math.cos(o),D=Math.sin(o);return i[0]=P*a+D*l,i[1]=P*g+D*x,i[2]=P*y+D*_,i[3]=P*c+D*m,i[4]=P*l-D*a,i[5]=P*x-D*g,i[6]=P*_-D*y,i[7]=P*m-D*c,e!==i&&(i[8]=e[8],i[9]=e[9],i[10]=e[10],i[11]=e[11],i[12]=e[12],i[13]=e[13],i[14]=e[14],i[15]=e[15]),i}function C(e,o,n){const i=n??new f(16);let a=e[0],g=e[1],y=e[2];const c=Math.sqrt(a*a+g*g+y*y);a/=c,g/=c,y/=c;const l=a*a,x=g*g,_=y*y,m=Math.cos(o),P=Math.sin(o),D=1-m;return i[0]=l+(1-l)*m,i[1]=a*g*D+y*P,i[2]=a*y*D-g*P,i[3]=0,i[4]=a*g*D-y*P,i[5]=x+(1-x)*m,i[6]=g*y*D+a*P,i[7]=0,i[8]=a*y*D+g*P,i[9]=g*y*D-a*P,i[10]=_+(1-_)*m,i[11]=0,i[12]=0,i[13]=0,i[14]=0,i[15]=1,i}const r=C;function p(e,o,n,i){const a=i??new f(16);let g=o[0],y=o[1],c=o[2];const l=Math.sqrt(g*g+y*y+c*c);g/=l,y/=l,c/=l;const x=g*g,_=y*y,m=c*c,P=Math.cos(n),D=Math.sin(n),S=1-P,O=x+(1-x)*P,B=g*y*S+c*D,R=g*c*S-y*D,L=g*y*S-c*D,H=_+(1-_)*P,re=y*c*S+g*D,pe=g*c*S+y*D,he=y*c*S-g*D,we=m+(1-m)*P,me=e[0],ye=e[1],Pe=e[2],Me=e[3],De=e[4],Ge=e[5],Be=e[6],Se=e[7],Re=e[8],Ie=e[9],Ee=e[10],Oe=e[11];return a[0]=O*me+B*De+R*Re,a[1]=O*ye+B*Ge+R*Ie,a[2]=O*Pe+B*Be+R*Ee,a[3]=O*Me+B*Se+R*Oe,a[4]=L*me+H*De+re*Re,a[5]=L*ye+H*Ge+re*Ie,a[6]=L*Pe+H*Be+re*Ee,a[7]=L*Me+H*Se+re*Oe,a[8]=pe*me+he*De+we*Re,a[9]=pe*ye+he*Ge+we*Ie,a[10]=pe*Pe+he*Be+we*Ee,a[11]=pe*Me+he*Se+we*Oe,e!==a&&(a[12]=e[12],a[13]=e[13],a[14]=e[14],a[15]=e[15]),a}const t=p;function s(e,o){const n=o??new f(16);return n[0]=e[0],n[1]=0,n[2]=0,n[3]=0,n[4]=0,n[5]=e[1],n[6]=0,n[7]=0,n[8]=0,n[9]=0,n[10]=e[2],n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,n}function u(e,o,n){const i=n??new f(16),a=o[0],g=o[1],y=o[2];return i[0]=a*e[0*4+0],i[1]=a*e[0*4+1],i[2]=a*e[0*4+2],i[3]=a*e[0*4+3],i[4]=g*e[1*4+0],i[5]=g*e[1*4+1],i[6]=g*e[1*4+2],i[7]=g*e[1*4+3],i[8]=y*e[2*4+0],i[9]=y*e[2*4+1],i[10]=y*e[2*4+2],i[11]=y*e[2*4+3],e!==i&&(i[12]=e[12],i[13]=e[13],i[14]=e[14],i[15]=e[15]),i}function d(e,o){const n=o??new f(16);return n[0]=e,n[1]=0,n[2]=0,n[3]=0,n[4]=0,n[5]=e,n[6]=0,n[7]=0,n[8]=0,n[9]=0,n[10]=e,n[11]=0,n[12]=0,n[13]=0,n[14]=0,n[15]=1,n}function w(e,o,n){const i=n??new f(16);return i[0]=o*e[0*4+0],i[1]=o*e[0*4+1],i[2]=o*e[0*4+2],i[3]=o*e[0*4+3],i[4]=o*e[1*4+0],i[5]=o*e[1*4+1],i[6]=o*e[1*4+2],i[7]=o*e[1*4+3],i[8]=o*e[2*4+0],i[9]=o*e[2*4+1],i[10]=o*e[2*4+2],i[11]=o*e[2*4+3],e!==i&&(i[12]=e[12],i[13]=e[13],i[14]=e[14],i[15]=e[15]),i}return{create:v,set:z,fromMat3:M,fromQuat:G,negate:I,copy:E,clone:F,equalsApproximately:K,equals:Z,identity:U,transpose:N,inverse:q,determinant:Q,invert:X,multiply:Y,mul:ee,setTranslation:se,getTranslation:ie,getAxis:le,setAxis:te,getScaling:oe,perspective:ue,perspectiveReverseZ:fe,ortho:de,frustum:ne,frustumReverseZ:xe,aim:J,cameraAim:W,lookAt:$,translation:ce,translate:ge,rotationX:j,rotateX:ae,rotationY:_e,rotateY:be,rotationZ:ve,rotateZ:Te,axisRotation:C,rotation:r,axisRotate:p,rotate:t,scaling:s,scale:u,uniformScaling:d,uniformScale:w}}const rn=new Map;function vn(f){let h=rn.get(f);return h||(h=_n(f),rn.set(f,h)),h}function wn(f){const h=Xe(f);function v(r,p,t,s){const u=new f(4);return r!==void 0&&(u[0]=r,p!==void 0&&(u[1]=p,t!==void 0&&(u[2]=t,s!==void 0&&(u[3]=s)))),u}const z=v;function M(r,p,t,s,u){const d=u??new f(4);return d[0]=r,d[1]=p,d[2]=t,d[3]=s,d}function G(r,p,t){const s=t??new f(4),u=p*.5,d=Math.sin(u);return s[0]=d*r[0],s[1]=d*r[1],s[2]=d*r[2],s[3]=Math.cos(u),s}function I(r,p){const t=p??h.create(3),s=Math.acos(r[3])*2,u=Math.sin(s*.5);return u>A?(t[0]=r[0]/u,t[1]=r[1]/u,t[2]=r[2]/u):(t[0]=1,t[1]=0,t[2]=0),{angle:s,axis:t}}function E(r,p){const t=ne(r,p);return Math.acos(2*t*t-1)}function F(r,p,t){const s=t??new f(4),u=r[0],d=r[1],w=r[2],e=r[3],o=p[0],n=p[1],i=p[2],a=p[3];return s[0]=u*a+e*o+d*i-w*n,s[1]=d*a+e*n+w*o-u*i,s[2]=w*a+e*i+u*n-d*o,s[3]=e*a-u*o-d*n-w*i,s}const K=F;function Z(r,p,t){const s=t??new f(4),u=p*.5,d=r[0],w=r[1],e=r[2],o=r[3],n=Math.sin(u),i=Math.cos(u);return s[0]=d*i+o*n,s[1]=w*i+e*n,s[2]=e*i-w*n,s[3]=o*i-d*n,s}function U(r,p,t){const s=t??new f(4),u=p*.5,d=r[0],w=r[1],e=r[2],o=r[3],n=Math.sin(u),i=Math.cos(u);return s[0]=d*i-e*n,s[1]=w*i+o*n,s[2]=e*i+d*n,s[3]=o*i-w*n,s}function N(r,p,t){const s=t??new f(4),u=p*.5,d=r[0],w=r[1],e=r[2],o=r[3],n=Math.sin(u),i=Math.cos(u);return s[0]=d*i+w*n,s[1]=w*i-d*n,s[2]=e*i+o*n,s[3]=o*i-e*n,s}function q(r,p,t,s){const u=s??new f(4),d=r[0],w=r[1],e=r[2],o=r[3];let n=p[0],i=p[1],a=p[2],g=p[3],y=d*n+w*i+e*a+o*g;y<0&&(y=-y,n=-n,i=-i,a=-a,g=-g);let c,l;if(1-y>A){const x=Math.acos(y),_=Math.sin(x);c=Math.sin((1-t)*x)/_,l=Math.sin(t*x)/_}else c=1-t,l=t;return u[0]=c*d+l*n,u[1]=c*w+l*i,u[2]=c*e+l*a,u[3]=c*o+l*g,u}function Q(r,p){const t=p??new f(4),s=r[0],u=r[1],d=r[2],w=r[3],e=s*s+u*u+d*d+w*w,o=e?1/e:0;return t[0]=-s*o,t[1]=-u*o,t[2]=-d*o,t[3]=w*o,t}function X(r,p){const t=p??new f(4);return t[0]=-r[0],t[1]=-r[1],t[2]=-r[2],t[3]=r[3],t}function Y(r,p){const t=p??new f(4),s=r[0]+r[5]+r[10];if(s>0){const u=Math.sqrt(s+1);t[3]=.5*u;const d=.5/u;t[0]=(r[6]-r[9])*d,t[1]=(r[8]-r[2])*d,t[2]=(r[1]-r[4])*d}else{let u=0;r[5]>r[0]&&(u=1),r[10]>r[u*4+u]&&(u=2);const d=(u+1)%3,w=(u+2)%3,e=Math.sqrt(r[u*4+u]-r[d*4+d]-r[w*4+w]+1);t[u]=.5*e;const o=.5/e;t[3]=(r[d*4+w]-r[w*4+d])*o,t[d]=(r[d*4+u]+r[u*4+d])*o,t[w]=(r[w*4+u]+r[u*4+w])*o}return t}function ee(r,p,t,s,u){const d=u??new f(4),w=r*.5,e=p*.5,o=t*.5,n=Math.sin(w),i=Math.cos(w),a=Math.sin(e),g=Math.cos(e),y=Math.sin(o),c=Math.cos(o);switch(s){case"xyz":d[0]=n*g*c+i*a*y,d[1]=i*a*c-n*g*y,d[2]=i*g*y+n*a*c,d[3]=i*g*c-n*a*y;break;case"xzy":d[0]=n*g*c-i*a*y,d[1]=i*a*c-n*g*y,d[2]=i*g*y+n*a*c,d[3]=i*g*c+n*a*y;break;case"yxz":d[0]=n*g*c+i*a*y,d[1]=i*a*c-n*g*y,d[2]=i*g*y-n*a*c,d[3]=i*g*c+n*a*y;break;case"yzx":d[0]=n*g*c+i*a*y,d[1]=i*a*c+n*g*y,d[2]=i*g*y-n*a*c,d[3]=i*g*c-n*a*y;break;case"zxy":d[0]=n*g*c-i*a*y,d[1]=i*a*c+n*g*y,d[2]=i*g*y+n*a*c,d[3]=i*g*c-n*a*y;break;case"zyx":d[0]=n*g*c-i*a*y,d[1]=i*a*c+n*g*y,d[2]=i*g*y-n*a*c,d[3]=i*g*c+n*a*y;break;default:throw new Error(`Unknown rotation order: ${s}`)}return d}function se(r,p){const t=p??new f(4);return t[0]=r[0],t[1]=r[1],t[2]=r[2],t[3]=r[3],t}const ie=se;function le(r,p,t){const s=t??new f(4);return s[0]=r[0]+p[0],s[1]=r[1]+p[1],s[2]=r[2]+p[2],s[3]=r[3]+p[3],s}function te(r,p,t){const s=t??new f(4);return s[0]=r[0]-p[0],s[1]=r[1]-p[1],s[2]=r[2]-p[2],s[3]=r[3]-p[3],s}const oe=te;function ue(r,p,t){const s=t??new f(4);return s[0]=r[0]*p,s[1]=r[1]*p,s[2]=r[2]*p,s[3]=r[3]*p,s}const fe=ue;function de(r,p,t){const s=t??new f(4);return s[0]=r[0]/p,s[1]=r[1]/p,s[2]=r[2]/p,s[3]=r[3]/p,s}function ne(r,p){return r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]*p[3]}function xe(r,p,t,s){const u=s??new f(4);return u[0]=r[0]+t*(p[0]-r[0]),u[1]=r[1]+t*(p[1]-r[1]),u[2]=r[2]+t*(p[2]-r[2]),u[3]=r[3]+t*(p[3]-r[3]),u}function k(r){const p=r[0],t=r[1],s=r[2],u=r[3];return Math.sqrt(p*p+t*t+s*s+u*u)}const V=k;function T(r){const p=r[0],t=r[1],s=r[2],u=r[3];return p*p+t*t+s*s+u*u}const J=T;function W(r,p){const t=p??new f(4),s=r[0],u=r[1],d=r[2],w=r[3],e=Math.sqrt(s*s+u*u+d*d+w*w);return e>1e-5?(t[0]=s/e,t[1]=u/e,t[2]=d/e,t[3]=w/e):(t[0]=0,t[1]=0,t[2]=0,t[3]=1),t}function $(r,p){return Math.abs(r[0]-p[0])<A&&Math.abs(r[1]-p[1])<A&&Math.abs(r[2]-p[2])<A&&Math.abs(r[3]-p[3])<A}function ce(r,p){return r[0]===p[0]&&r[1]===p[1]&&r[2]===p[2]&&r[3]===p[3]}function ge(r){const p=r??new f(4);return p[0]=0,p[1]=0,p[2]=0,p[3]=1,p}const j=h.create(),ae=h.create(),_e=h.create();function be(r,p,t){const s=t??new f(4),u=h.dot(r,p);return u<-.999999?(h.cross(ae,r,j),h.len(j)<1e-6&&h.cross(_e,r,j),h.normalize(j,j),G(j,Math.PI,s),s):u>.999999?(s[0]=0,s[1]=0,s[2]=0,s[3]=1,s):(h.cross(r,p,j),s[0]=j[0],s[1]=j[1],s[2]=j[2],s[3]=1+u,W(s,s))}const ve=new f(4),Te=new f(4);function C(r,p,t,s,u,d){const w=d??new f(4);return q(r,s,u,ve),q(p,t,u,Te),q(ve,Te,2*u*(1-u),w),w}return{create:v,fromValues:z,set:M,fromAxisAngle:G,toAxisAngle:I,angle:E,multiply:F,mul:K,rotateX:Z,rotateY:U,rotateZ:N,slerp:q,inverse:Q,conjugate:X,fromMat:Y,fromEuler:ee,copy:se,clone:ie,add:le,subtract:te,sub:oe,mulScalar:ue,scale:fe,divScalar:de,dot:ne,lerp:xe,length:k,len:V,lengthSq:T,lenSq:J,normalize:W,equalsApproximately:$,equals:ce,identity:ge,rotationTo:be,sqlerp:C}}const sn=new Map;function mn(f){let h=sn.get(f);return h||(h=wn(f),sn.set(f,h)),h}function yn(f){function h(t,s,u,d){const w=new f(4);return t!==void 0&&(w[0]=t,s!==void 0&&(w[1]=s,u!==void 0&&(w[2]=u,d!==void 0&&(w[3]=d)))),w}const v=h;function z(t,s,u,d,w){const e=w??new f(4);return e[0]=t,e[1]=s,e[2]=u,e[3]=d,e}function M(t,s){const u=s??new f(4);return u[0]=Math.ceil(t[0]),u[1]=Math.ceil(t[1]),u[2]=Math.ceil(t[2]),u[3]=Math.ceil(t[3]),u}function G(t,s){const u=s??new f(4);return u[0]=Math.floor(t[0]),u[1]=Math.floor(t[1]),u[2]=Math.floor(t[2]),u[3]=Math.floor(t[3]),u}function I(t,s){const u=s??new f(4);return u[0]=Math.round(t[0]),u[1]=Math.round(t[1]),u[2]=Math.round(t[2]),u[3]=Math.round(t[3]),u}function E(t,s=0,u=1,d){const w=d??new f(4);return w[0]=Math.min(u,Math.max(s,t[0])),w[1]=Math.min(u,Math.max(s,t[1])),w[2]=Math.min(u,Math.max(s,t[2])),w[3]=Math.min(u,Math.max(s,t[3])),w}function F(t,s,u){const d=u??new f(4);return d[0]=t[0]+s[0],d[1]=t[1]+s[1],d[2]=t[2]+s[2],d[3]=t[3]+s[3],d}function K(t,s,u,d){const w=d??new f(4);return w[0]=t[0]+s[0]*u,w[1]=t[1]+s[1]*u,w[2]=t[2]+s[2]*u,w[3]=t[3]+s[3]*u,w}function Z(t,s,u){const d=u??new f(4);return d[0]=t[0]-s[0],d[1]=t[1]-s[1],d[2]=t[2]-s[2],d[3]=t[3]-s[3],d}const U=Z;function N(t,s){return Math.abs(t[0]-s[0])<A&&Math.abs(t[1]-s[1])<A&&Math.abs(t[2]-s[2])<A&&Math.abs(t[3]-s[3])<A}function q(t,s){return t[0]===s[0]&&t[1]===s[1]&&t[2]===s[2]&&t[3]===s[3]}function Q(t,s,u,d){const w=d??new f(4);return w[0]=t[0]+u*(s[0]-t[0]),w[1]=t[1]+u*(s[1]-t[1]),w[2]=t[2]+u*(s[2]-t[2]),w[3]=t[3]+u*(s[3]-t[3]),w}function X(t,s,u,d){const w=d??new f(4);return w[0]=t[0]+u[0]*(s[0]-t[0]),w[1]=t[1]+u[1]*(s[1]-t[1]),w[2]=t[2]+u[2]*(s[2]-t[2]),w[3]=t[3]+u[3]*(s[3]-t[3]),w}function Y(t,s,u){const d=u??new f(4);return d[0]=Math.max(t[0],s[0]),d[1]=Math.max(t[1],s[1]),d[2]=Math.max(t[2],s[2]),d[3]=Math.max(t[3],s[3]),d}function ee(t,s,u){const d=u??new f(4);return d[0]=Math.min(t[0],s[0]),d[1]=Math.min(t[1],s[1]),d[2]=Math.min(t[2],s[2]),d[3]=Math.min(t[3],s[3]),d}function se(t,s,u){const d=u??new f(4);return d[0]=t[0]*s,d[1]=t[1]*s,d[2]=t[2]*s,d[3]=t[3]*s,d}const ie=se;function le(t,s,u){const d=u??new f(4);return d[0]=t[0]/s,d[1]=t[1]/s,d[2]=t[2]/s,d[3]=t[3]/s,d}function te(t,s){const u=s??new f(4);return u[0]=1/t[0],u[1]=1/t[1],u[2]=1/t[2],u[3]=1/t[3],u}const oe=te;function ue(t,s){return t[0]*s[0]+t[1]*s[1]+t[2]*s[2]+t[3]*s[3]}function fe(t){const s=t[0],u=t[1],d=t[2],w=t[3];return Math.sqrt(s*s+u*u+d*d+w*w)}const de=fe;function ne(t){const s=t[0],u=t[1],d=t[2],w=t[3];return s*s+u*u+d*d+w*w}const xe=ne;function k(t,s){const u=t[0]-s[0],d=t[1]-s[1],w=t[2]-s[2],e=t[3]-s[3];return Math.sqrt(u*u+d*d+w*w+e*e)}const V=k;function T(t,s){const u=t[0]-s[0],d=t[1]-s[1],w=t[2]-s[2],e=t[3]-s[3];return u*u+d*d+w*w+e*e}const J=T;function W(t,s){const u=s??new f(4),d=t[0],w=t[1],e=t[2],o=t[3],n=Math.sqrt(d*d+w*w+e*e+o*o);return n>1e-5?(u[0]=d/n,u[1]=w/n,u[2]=e/n,u[3]=o/n):(u[0]=0,u[1]=0,u[2]=0,u[3]=0),u}function $(t,s){const u=s??new f(4);return u[0]=-t[0],u[1]=-t[1],u[2]=-t[2],u[3]=-t[3],u}function ce(t,s){const u=s??new f(4);return u[0]=t[0],u[1]=t[1],u[2]=t[2],u[3]=t[3],u}const ge=ce;function j(t,s,u){const d=u??new f(4);return d[0]=t[0]*s[0],d[1]=t[1]*s[1],d[2]=t[2]*s[2],d[3]=t[3]*s[3],d}const ae=j;function _e(t,s,u){const d=u??new f(4);return d[0]=t[0]/s[0],d[1]=t[1]/s[1],d[2]=t[2]/s[2],d[3]=t[3]/s[3],d}const be=_e;function ve(t){const s=t??new f(4);return s[0]=0,s[1]=0,s[2]=0,s[3]=0,s}function Te(t,s,u){const d=u??new f(4),w=t[0],e=t[1],o=t[2],n=t[3];return d[0]=s[0]*w+s[4]*e+s[8]*o+s[12]*n,d[1]=s[1]*w+s[5]*e+s[9]*o+s[13]*n,d[2]=s[2]*w+s[6]*e+s[10]*o+s[14]*n,d[3]=s[3]*w+s[7]*e+s[11]*o+s[15]*n,d}function C(t,s,u){const d=u??new f(4);return W(t,d),se(d,s,d)}function r(t,s,u){const d=u??new f(4);return fe(t)>s?C(t,s,d):ce(t,d)}function p(t,s,u){const d=u??new f(4);return Q(t,s,.5,d)}return{create:h,fromValues:v,set:z,ceil:M,floor:G,round:I,clamp:E,add:F,addScaled:K,subtract:Z,sub:U,equalsApproximately:N,equals:q,lerp:Q,lerpV:X,max:Y,min:ee,mulScalar:se,scale:ie,divScalar:le,inverse:te,invert:oe,dot:ue,length:fe,len:de,lengthSq:ne,lenSq:xe,distance:k,dist:V,distanceSq:T,distSq:J,normalize:W,negate:$,copy:ce,clone:ge,multiply:j,mul:ae,divide:_e,div:be,zero:ve,transformMat4:Te,setLength:C,truncate:r,midpoint:p}}const on=new Map;function Pn(f){let h=on.get(f);return h||(h=yn(f),on.set(f,h)),h}function Ze(f,h,v,z,M,G){return{mat3:gn(f),mat4:vn(h),quat:mn(v),vec2:an(z),vec3:Xe(M),vec4:Pn(G)}}const{mat3:Jn,mat4:ke,quat:et,vec2:nt,vec3:tt,vec4:it}=Ze(Float32Array,Float32Array,Float32Array,Float32Array,Float32Array,Float32Array);Ze(Float64Array,Float64Array,Float64Array,Float64Array,Float64Array,Float64Array);Ze(dn,Array,Array,Array,Array,Array);const Ue=new ArrayBuffer(272),Fe={texel_size:new Float32Array(Ue,0,2),sphere_size:new Float32Array(Ue,8,2),inv_projection_matrix:new Float32Array(Ue,16,16),projection_matrix:new Float32Array(Ue,80,16),view_matrix:new Float32Array(Ue,144,16),inv_view_matrix:new Float32Array(Ue,208,16)},Ae=2e5;class Mn{constructor(h){b(this,"isDragging");b(this,"prevX");b(this,"prevY");b(this,"currentXtheta");b(this,"currentYtheta");b(this,"maxYTheta");b(this,"minYTheta");b(this,"sensitivity");b(this,"currentDistance");b(this,"maxDistance");b(this,"minDistance");b(this,"target");b(this,"fov");b(this,"zoomRate");h.addEventListener("mousedown",v=>{this.isDragging=!0,this.prevX=v.clientX,this.prevY=v.clientY}),h.addEventListener("wheel",v=>{v.preventDefault();var z=v.deltaY;this.currentDistance+=(z>0?1:-1)*this.zoomRate,this.currentDistance<this.minDistance&&(this.currentDistance=this.minDistance),this.currentDistance>this.maxDistance&&(this.currentDistance=this.maxDistance),this.recalculateView()}),h.addEventListener("mousemove",v=>{if(this.isDragging){const z=v.clientX,M=v.clientY,G=this.prevX-z,I=this.prevY-M;this.currentXtheta+=this.sensitivity*G,this.currentYtheta+=this.sensitivity*I,this.currentYtheta>this.maxYTheta&&(this.currentYtheta=this.maxYTheta),this.currentYtheta<this.minYTheta&&(this.currentYtheta=this.minYTheta),this.prevX=z,this.prevY=M,this.recalculateView()}}),h.addEventListener("mouseup",()=>{this.isDragging&&(this.isDragging=!1)})}reset(h,v,z,M,G){this.isDragging=!1,this.prevX=0,this.prevY=0,this.currentXtheta=Math.PI/4*1,this.currentYtheta=-Math.PI/12,this.maxYTheta=0,this.minYTheta=-.99*Math.PI/2,this.sensitivity=.005,this.currentDistance=v,this.maxDistance=2*this.currentDistance,this.minDistance=.3*this.currentDistance,this.target=z,this.fov=M,this.zoomRate=G;const I=h.clientWidth/h.clientHeight,E=ke.perspective(M,I,.1,500);Fe.projection_matrix.set(E),Fe.inv_projection_matrix.set(ke.inverse(E)),this.recalculateView()}recalculateView(){var h=ke.identity();ke.translate(h,this.target,h),ke.rotateY(h,this.currentXtheta,h),ke.rotateX(h,this.currentYtheta,h),ke.translate(h,[0,0,this.currentDistance],h);var v=ke.multiply(h,[0,0,0,1]);const z=ke.lookAt([v[0],v[1],v[2]],this.target,[0,1,0]);Fe.view_matrix.set(z),Fe.inv_view_matrix.set(ke.inverse(z))}}var Dn=`struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        cells[id.x].mass = 0;
        cells[id.x].vx = 0;
        cells[id.x].vy = 0;
        cells[id.x].vz = 0;
    }
}`,zn=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
}
struct Cell {
    vx: atomic<i32>, 
    vy: atomic<i32>, 
    vz: atomic<i32>, 
    mass: atomic<i32>, 
}

override fixed_point_multiplier: f32; 

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

@compute @workgroup_size(64)
fn p2g_1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cell_idx: vec3f = floor(particle.position);
        let cell_diff: vec3f = particle.position - (cell_idx + 0.5f);
        weights[0] = 0.5f * (0.5f - cell_diff) * (0.5f - cell_diff);
        weights[1] = 0.75f - cell_diff * cell_diff;
        weights[2] = 0.5f * (0.5f + cell_diff) * (0.5f + cell_diff);

        let C: mat3x3f = particle.C;

        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cell_x: vec3f = vec3f(
                            cell_idx.x + f32(gx) - 1., 
                            cell_idx.y + f32(gy) - 1.,
                            cell_idx.z + f32(gz) - 1.  
                        );
                    let cell_dist = (cell_x + 0.5f) - particle.position;

                    let Q: vec3f = C * cell_dist;

                    let mass_contrib: f32 = weight * 1.0; 
                    let vel_contrib: vec3f = mass_contrib * (particle.v + Q);
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    atomicAdd(&cells[cell_index].mass, encodeFixedPoint(mass_contrib));
                    atomicAdd(&cells[cell_index].vx, encodeFixedPoint(vel_contrib.x));
                    atomicAdd(&cells[cell_index].vy, encodeFixedPoint(vel_contrib.y));
                    atomicAdd(&cells[cell_index].vz, encodeFixedPoint(vel_contrib.z));
                }
            }
        }
    }
}`,bn=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
}
struct Cell {
    vx: atomic<i32>, 
    vy: atomic<i32>, 
    vz: atomic<i32>, 
    mass: i32, 
}

override fixed_point_multiplier: f32; 
override stiffness: f32;
override rest_density: f32;
override dynamic_viscosity: f32;
override dt: f32;

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}
fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

@compute @workgroup_size(64)
fn p2g_2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cell_idx: vec3f = floor(particle.position);
        let cell_diff: vec3f = particle.position - (cell_idx + 0.5f);
        weights[0] = 0.5f * (0.5f - cell_diff) * (0.5f - cell_diff);
        weights[1] = 0.75f - cell_diff * cell_diff;
        weights[2] = 0.5f * (0.5f + cell_diff) * (0.5f + cell_diff);

        var density: f32 = 0.;
        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cell_x: vec3f = vec3f(
                            cell_idx.x + f32(gx) - 1., 
                            cell_idx.y + f32(gy) - 1.,
                            cell_idx.z + f32(gz) - 1.  
                        );
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    density += decodeFixedPoint(cells[cell_index].mass) * weight;
                }
            }
        }

        let volume: f32 = 1.0 / density; 

        let pressure: f32 = max(-0.0, stiffness * (pow(density / rest_density, 5.) - 1));

        var stress: mat3x3f = mat3x3f(-pressure, 0, 0, 0, -pressure, 0, 0, 0, -pressure);
        let dudv: mat3x3f = particle.C;
        let strain: mat3x3f = dudv + transpose(dudv);
        stress += dynamic_viscosity * strain;

        let eq_16_term0 = -volume * 4 * stress * dt;

        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cell_x: vec3f = vec3f(
                            cell_idx.x + f32(gx) - 1., 
                            cell_idx.y + f32(gy) - 1.,
                            cell_idx.z + f32(gz) - 1.  
                        );
                    let cell_dist = (cell_x + 0.5f) - particle.position;
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    let momentum: vec3f = eq_16_term0 * weight * cell_dist;
                    atomicAdd(&cells[cell_index].vx, encodeFixedPoint(momentum.x));
                    atomicAdd(&cells[cell_index].vy, encodeFixedPoint(momentum.y));
                    atomicAdd(&cells[cell_index].vz, encodeFixedPoint(momentum.z));
                }
            }
        }
    }
}`,Gn=`struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}

override fixed_point_multiplier: f32; 
override dt: f32; 

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}
fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}

@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        if (cells[id.x].mass > 0) { 
            var float_v: vec3f = vec3f(
                decodeFixedPoint(cells[id.x].vx), 
                decodeFixedPoint(cells[id.x].vy), 
                decodeFixedPoint(cells[id.x].vz)
            );
            float_v /= decodeFixedPoint(cells[id.x].mass);
            cells[id.x].vx = encodeFixedPoint(float_v.x);
            cells[id.x].vy = encodeFixedPoint(float_v.y + -0.3 * dt);
            cells[id.x].vz = encodeFixedPoint(float_v.z);

            var x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
            var y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
            var z: i32 = i32(id.x) % i32(init_box_size.z);
            
            if (x < 2 || x > i32(ceil(real_box_size.x) - 3)) { cells[id.x].vx = 0; } 
            if (y < 2 || y > i32(ceil(real_box_size.y) - 3)) { cells[id.x].vy = 0; }
            if (z < 2 || z > i32(ceil(real_box_size.z) - 3)) { cells[id.x].vz = 0; }
        }
    }
}`,Bn=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
}
struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}

override fixed_point_multiplier: f32; 
override dt: f32; 

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> real_box_size: vec3f;
@group(0) @binding(3) var<uniform> init_box_size: vec3f;

fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        particles[id.x].v = vec3f(0.);
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cell_idx: vec3f = floor(particle.position);
        let cell_diff: vec3f = particle.position - (cell_idx + 0.5f);
        weights[0] = 0.5f * (0.5f - cell_diff) * (0.5f - cell_diff);
        weights[1] = 0.75f - cell_diff * cell_diff;
        weights[2] = 0.5f * (0.5f + cell_diff) * (0.5f + cell_diff);

        var B: mat3x3f = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cell_x: vec3f = vec3f(
                        cell_idx.x + f32(gx) - 1., 
                        cell_idx.y + f32(gy) - 1.,
                        cell_idx.z + f32(gz) - 1.  
                    );
                    let cell_dist: vec3f = (cell_x + 0.5f) - particle.position;
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    let weighted_velocity: vec3f = vec3f(
                        decodeFixedPoint(cells[cell_index].vx), 
                        decodeFixedPoint(cells[cell_index].vy), 
                        decodeFixedPoint(cells[cell_index].vz)
                    ) * weight;
                    let term: mat3x3f = mat3x3f(
                        weighted_velocity * cell_dist.x, 
                        weighted_velocity * cell_dist.y, 
                        weighted_velocity * cell_dist.z
                    );

                    B += term;

                    particles[id.x].v += weighted_velocity;
                }
            }
        }

        particles[id.x].C = B * 4.0f;
        particles[id.x].position += particles[id.x].v * dt;
        particles[id.x].position = vec3f(
            clamp(particles[id.x].position.x, 1., real_box_size.x - 2.), 
            clamp(particles[id.x].position.y, 1., real_box_size.y - 2.), 
            clamp(particles[id.x].position.z, 1., real_box_size.z - 2.)
        );
        
        let k = 3.0;
        let wall_stiffness = 0.3;
        let x_n: vec3f = particles[id.x].position + particles[id.x].v * dt * k;
        let wall_min: vec3f = vec3f(3.);
        let wall_max: vec3f = real_box_size - 4.;
        if (x_n.x < wall_min.x) { particles[id.x].v.x += wall_stiffness * (wall_min.x - x_n.x); }
        if (x_n.x > wall_max.x) { particles[id.x].v.x += wall_stiffness * (wall_max.x - x_n.x); }
        if (x_n.y < wall_min.y) { particles[id.x].v.y += wall_stiffness * (wall_min.y - x_n.y); }
        if (x_n.y > wall_max.y) { particles[id.x].v.y += wall_stiffness * (wall_max.y - x_n.y); }
        if (x_n.z < wall_min.z) { particles[id.x].v.z += wall_stiffness * (wall_min.z - x_n.z); }
        if (x_n.z > wall_max.z) { particles[id.x].v.z += wall_stiffness * (wall_max.z - x_n.z); }
    }
}`,Sn=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
}

struct PosVel {
    position: vec3f, 
    v: vec3f, 
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> posvel: array<PosVel>;

@compute @workgroup_size(64)
fn copyPosition(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) { 
        posvel[id.x].position = particles[id.x].position;
        posvel[id.x].v = particles[id.x].v;
    }
}`;const Ye=80;class Rn{constructor(h,v,z,M){b(this,"max_x_grids",64);b(this,"max_y_grids",64);b(this,"max_z_grids",64);b(this,"cellStructSize",16);b(this,"realBoxSizeBuffer");b(this,"initBoxSizeBuffer");b(this,"numParticles",0);b(this,"gridCount",0);b(this,"clearGridPipeline");b(this,"p2g1Pipeline");b(this,"p2g2Pipeline");b(this,"updateGridPipeline");b(this,"g2pPipeline");b(this,"copyPositionPipeline");b(this,"clearGridBindGroup");b(this,"p2g1BindGroup");b(this,"p2g2BindGroup");b(this,"updateGridBindGroup");b(this,"g2pBindGroup");b(this,"copyPositionBindGroup");b(this,"particleBuffer");b(this,"device");b(this,"renderDiameter");this.device=M,this.renderDiameter=z;const G=M.createShaderModule({code:Dn}),I=M.createShaderModule({code:zn}),E=M.createShaderModule({code:bn}),F=M.createShaderModule({code:Gn}),K=M.createShaderModule({code:Bn}),Z=M.createShaderModule({code:Sn}),U={stiffness:3,restDensity:4,dynamic_viscosity:.1,dt:.2,fixed_point_multiplier:1e7};this.clearGridPipeline=M.createComputePipeline({label:"clear grid pipeline",layout:"auto",compute:{module:G}}),this.p2g1Pipeline=M.createComputePipeline({label:"p2g 1 pipeline",layout:"auto",compute:{module:I,constants:{fixed_point_multiplier:U.fixed_point_multiplier}}}),this.p2g2Pipeline=M.createComputePipeline({label:"p2g 2 pipeline",layout:"auto",compute:{module:E,constants:{fixed_point_multiplier:U.fixed_point_multiplier,stiffness:U.stiffness,rest_density:U.restDensity,dynamic_viscosity:U.dynamic_viscosity,dt:U.dt}}}),this.updateGridPipeline=M.createComputePipeline({label:"update grid pipeline",layout:"auto",compute:{module:F,constants:{fixed_point_multiplier:U.fixed_point_multiplier,dt:U.dt}}}),this.g2pPipeline=M.createComputePipeline({label:"g2p pipeline",layout:"auto",compute:{module:K,constants:{fixed_point_multiplier:U.fixed_point_multiplier,dt:U.dt}}}),this.copyPositionPipeline=M.createComputePipeline({label:"copy position pipeline",layout:"auto",compute:{module:Z}});const N=this.max_x_grids*this.max_y_grids*this.max_z_grids,q=new ArrayBuffer(12),Q=new ArrayBuffer(12),X=M.createBuffer({label:"cells buffer",size:this.cellStructSize*N,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.realBoxSizeBuffer=M.createBuffer({label:"real box size buffer",size:q.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.initBoxSizeBuffer=M.createBuffer({label:"init box size buffer",size:Q.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),M.queue.writeBuffer(this.initBoxSizeBuffer,0,Q),M.queue.writeBuffer(this.realBoxSizeBuffer,0,q),this.clearGridBindGroup=M.createBindGroup({layout:this.clearGridPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:X}}]}),this.p2g1BindGroup=M.createBindGroup({layout:this.p2g1Pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:X}},{binding:2,resource:{buffer:this.initBoxSizeBuffer}}]}),this.p2g2BindGroup=M.createBindGroup({layout:this.p2g2Pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:X}},{binding:2,resource:{buffer:this.initBoxSizeBuffer}}]}),this.updateGridBindGroup=M.createBindGroup({layout:this.updateGridPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:X}},{binding:1,resource:{buffer:this.realBoxSizeBuffer}},{binding:2,resource:{buffer:this.initBoxSizeBuffer}}]}),this.g2pBindGroup=M.createBindGroup({layout:this.g2pPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:X}},{binding:2,resource:{buffer:this.realBoxSizeBuffer}},{binding:3,resource:{buffer:this.initBoxSizeBuffer}}]}),this.copyPositionBindGroup=M.createBindGroup({layout:this.copyPositionPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:v}}]}),this.particleBuffer=h}initDambreak(h,v){let z=new ArrayBuffer(Ye*Ae);const M=.65;this.numParticles=0;for(let F=0;F<h[1]*.8&&this.numParticles<v;F+=M)for(let K=3;K<h[0]-4&&this.numParticles<v;K+=M)for(let Z=3;Z<h[2]/2&&this.numParticles<v;Z+=M){const U=Ye*this.numParticles,N={position:new Float32Array(z,U+0,3),v:new Float32Array(z,U+16,3),C:new Float32Array(z,U+32,12)},q=2*Math.random();N.position.set([K+q,F+q,Z+q]),this.numParticles++}let G=new ArrayBuffer(Ye*this.numParticles);const I=new Uint8Array(z),E=new Uint8Array(G);return E.set(I.subarray(0,E.length)),G}reset(h,v){Fe.sphere_size.set([this.renderDiameter]);const z=this.initDambreak(v,h),M=this.max_x_grids*this.max_y_grids*this.max_z_grids;if(this.gridCount=Math.ceil(v[0])*Math.ceil(v[1])*Math.ceil(v[2]),this.gridCount>M)throw new Error("gridCount should be equal to or less than maxGridCount");const G=new ArrayBuffer(12),I=new Float32Array(G),E=new ArrayBuffer(12);new Float32Array(E).set(v),I.set(v),this.device.queue.writeBuffer(this.initBoxSizeBuffer,0,E),this.device.queue.writeBuffer(this.realBoxSizeBuffer,0,G),this.device.queue.writeBuffer(this.particleBuffer,0,z),console.log(this.numParticles)}execute(h){const v=h.beginComputePass();for(let z=0;z<2;z++)v.setBindGroup(0,this.clearGridBindGroup),v.setPipeline(this.clearGridPipeline),v.dispatchWorkgroups(Math.ceil(this.gridCount/64)),v.setBindGroup(0,this.p2g1BindGroup),v.setPipeline(this.p2g1Pipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.p2g2BindGroup),v.setPipeline(this.p2g2Pipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.updateGridBindGroup),v.setPipeline(this.updateGridPipeline),v.dispatchWorkgroups(Math.ceil(this.gridCount/64)),v.setBindGroup(0,this.g2pBindGroup),v.setPipeline(this.g2pPipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.copyPositionBindGroup),v.setPipeline(this.copyPositionPipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64));v.end()}changeBoxSize(h){const v=new ArrayBuffer(12),z=new Float32Array(v);z.set(h),this.device.queue.writeBuffer(this.realBoxSizeBuffer,0,z)}}var Tn=`@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<u32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
    
    if (id.x < arrayLength(&cellParticleCount)) {
        cellParticleCount[id.x] = 0u;
    }
}`,In=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct Environment {
    xGrids: i32, 
    yGrids: i32, 
    zGrids: i32, 
    cellSize: f32, 
    xHalf: f32, 
    yHalf: f32, 
    zHalf: f32, 
    offset: f32, 
}

struct SPHParams {
    mass: f32, 
    kernelRadius: f32, 
    kernelRadiusPow2: f32, 
    kernelRadiusPow5: f32, 
    kernelRadiusPow6: f32,  
    kernelRadiusPow9: f32, 
    dt: f32, 
    stiffness: f32, 
    nearStiffness: f32, 
    restDensity: f32, 
    viscosity: f32, 
    n: u32
}

@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> particleCellOffset : array<u32>;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(3) var<uniform> env: Environment;
@group(0) @binding(4) var<uniform> params: SPHParams;

fn cellId(position: vec3f) -> i32 {
    let xi: i32 = i32(floor((position.x + env.xHalf + env.offset) / env.cellSize));
    let yi: i32 = i32(floor((position.y + env.yHalf + env.offset) / env.cellSize));
    let zi: i32 = i32(floor((position.z + env.zHalf + env.offset) / env.cellSize));

    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
  if (id.x < params.n)
  {
    let cellID: i32 = cellId(particles[id.x].position);
    
    if (cellID < env.xGrids * env.yGrids * env.zGrids) { 
      particleCellOffset[id.x] = atomicAdd(&cellParticleCount[cellID], 1u);
    }
  }
}`,En=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct SPHParams {
    mass: f32, 
    kernelRadius: f32, 
    kernelRadiusPow2: f32, 
    kernelRadiusPow5: f32, 
    kernelRadiusPow6: f32,  
    kernelRadiusPow9: f32, 
    dt: f32, 
    stiffness: f32, 
    nearStiffness: f32, 
    restDensity: f32, 
    viscosity: f32, 
    n: u32
}

@group(0) @binding(0) var<storage, read> sourceParticles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> targetParticles: array<Particle>;
@group(0) @binding(2) var<storage, read> cellParticleCount : array<u32>;
@group(0) @binding(3) var<storage, read> particleCellOffset : array<u32>;
@group(0) @binding(4) var<uniform> env : Environment;
@group(0) @binding(5) var<uniform> params : SPHParams;

struct Environment {
    xGrids: i32, 
    yGrids: i32, 
    zGrids: i32, 
    cellSize: f32, 
    xHalf: f32, 
    yHalf: f32, 
    zHalf: f32, 
    offset: f32, 
}

fn cellId(position: vec3f) -> i32 {
    let xi: i32 = i32(floor((position.x + env.xHalf + env.offset) / env.cellSize));
    let yi: i32 = i32(floor((position.y + env.yHalf + env.offset) / env.cellSize));
    let zi: i32 = i32(floor((position.z + env.zHalf + env.offset) / env.cellSize));

    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x < params.n) {
        let cellId: i32 = cellId(sourceParticles[id.x].position);
        
        if (cellId < env.xGrids * env.yGrids * env.zGrids) {
            let targetIndex = cellParticleCount[cellId + 1] - particleCellOffset[id.x] - 1;
            if (targetIndex < params.n) {
                targetParticles[targetIndex] = sourceParticles[id.x];
            }
        }
    }
}`,On=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct Environment {
    xGrids: i32, 
    yGrids: i32, 
    zGrids: i32, 
    cellSize: f32, 
    xHalf: f32, 
    yHalf: f32, 
    zHalf: f32, 
    offset: f32, 
}

struct SPHParams {
    mass: f32, 
    kernelRadius: f32, 
    kernelRadiusPow2: f32, 
    kernelRadiusPow5: f32, 
    kernelRadiusPow6: f32,  
    kernelRadiusPow9: f32, 
    dt: f32, 
    stiffness: f32, 
    nearStiffness: f32, 
    restDensity: f32, 
    viscosity: f32, 
    n: u32
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> sortedParticles: array<Particle>;
@group(0) @binding(2) var<storage, read> prefixSum: array<u32>;
@group(0) @binding(3) var<uniform> env: Environment;
@group(0) @binding(4) var<uniform> params: SPHParams;

fn nearDensityKernel(r: f32) -> f32 {
    let scale = 15.0 / (3.1415926535 * params.kernelRadiusPow6);
    let d = params.kernelRadius - r;
    return scale * d * d * d;
}

fn densityKernel(r: f32) -> f32 {
    let scale = 315.0 / (64. * 3.1415926535 * params.kernelRadiusPow9);
    let dd = params.kernelRadiusPow2 - r * r;
    return scale * dd * dd * dd;
}

fn cellPosition(v: vec3f) -> vec3i {
    let xi = i32(floor((v.x + env.xHalf + env.offset) / env.cellSize));
    let yi = i32(floor((v.y + env.yHalf + env.offset) / env.cellSize));
    let zi = i32(floor((v.z + env.zHalf + env.offset) / env.cellSize));
    return vec3i(xi, yi, zi);
}

fn cellNumberFromId(xi: i32, yi: i32, zi: i32) -> i32 {
    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

@compute @workgroup_size(64)
fn computeDensity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.n) {
        particles[id.x].density = 0.0;
        particles[id.x].nearDensity = 0.0;
        let pos_i = particles[id.x].position;
        let n = params.n;

        let v = cellPosition(pos_i);
        if (v.x < env.xGrids && 0 <= v.x && 
            v.y < env.yGrids && 0 <= v.y && 
            v.z < env.zGrids && 0 <= v.z) 
        {
            for (var dz = max(-1, -v.z); dz <= min(1, env.zGrids - v.z - 1); dz++) {
                for (var dy = max(-1, -v.y); dy <= min(1, env.yGrids - v.y - 1); dy++) {
                    let dxMin = max(-1, -v.x);
                    let dxMax = min(1, env.xGrids - v.x - 1);
                    let startCellNum = cellNumberFromId(v.x + dxMin, v.y + dy, v.z + dz);
                    let endCellNum = cellNumberFromId(v.x + dxMax, v.y + dy, v.z + dz);
                    let start = prefixSum[startCellNum];
                    let end = prefixSum[endCellNum + 1];
                    for (var j = start; j < end; j++) {
                        let pos_j = sortedParticles[j].position;
                        let r2 = dot(pos_i - pos_j, pos_i - pos_j);
                        if (r2 < params.kernelRadiusPow2) {
                            particles[id.x].density += params.mass * densityKernel(sqrt(r2));
                            particles[id.x].nearDensity += params.mass * nearDensityKernel(sqrt(r2));
                        }
                    }
                }
            }
        }

        
        
        
        
        
        
        
        
        
    }
}`,kn=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct Environment {
    xGrids: i32, 
    yGrids: i32, 
    zGrids: i32, 
    cellSize: f32, 
    xHalf: f32, 
    yHalf: f32, 
    zHalf: f32, 
    offset: f32, 
}

struct SPHParams {
    mass: f32, 
    kernelRadius: f32, 
    kernelRadiusPow2: f32, 
    kernelRadiusPow5: f32, 
    kernelRadiusPow6: f32,  
    kernelRadiusPow9: f32, 
    dt: f32, 
    stiffness: f32, 
    nearStiffness: f32, 
    restDensity: f32, 
    viscosity: f32, 
    n: u32
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> sortedParticles: array<Particle>;
@group(0) @binding(2) var<storage, read> prefixSum: array<u32>;
@group(0) @binding(3) var<uniform> env: Environment;
@group(0) @binding(4) var<uniform> params: SPHParams;

fn densityKernelGradient(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * params.kernelRadiusPow6); 
    let d = params.kernelRadius - r;
    return scale * d * d;
}

fn nearDensityKernelGradient(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * params.kernelRadiusPow5); 
    let a = params.kernelRadiusPow9;
    let d = params.kernelRadius - r;
    return scale * d * d;
}

fn viscosityKernelLaplacian(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * params.kernelRadiusPow6);
    
    let d = params.kernelRadius - r;
    return scale * d;
}

fn cellPosition(v: vec3f) -> vec3i {
    let xi = i32(floor((v.x + env.xHalf + env.offset) / env.cellSize));
    let yi = i32(floor((v.y + env.yHalf + env.offset) / env.cellSize));
    let zi = i32(floor((v.z + env.zHalf + env.offset) / env.cellSize));
    return vec3i(xi, yi, zi);
}

fn cellNumberFromId(xi: i32, yi: i32, zi: i32) -> i32 {
    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

@compute @workgroup_size(64)
fn computeForce(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.n) {
        let n = params.n;
        let density_i = particles[id.x].density;
        let nearDensity_i = particles[id.x].nearDensity;
        let pos_i = particles[id.x].position;
        var fPress = vec3(0.0, 0.0, 0.0);
        var fVisc = vec3(0.0, 0.0, 0.0);

        let v = cellPosition(pos_i);
        if (v.x < env.xGrids && 0 <= v.x && 
            v.y < env.yGrids && 0 <= v.y && 
            v.z < env.zGrids && 0 <= v.z) 
        {
            if (v.x < env.xGrids && v.y < env.yGrids && v.z < env.zGrids) {
                for (var dz = max(-1, -v.z); dz <= min(1, env.zGrids - v.z - 1); dz++) {
                    for (var dy = max(-1, -v.y); dy <= min(1, env.yGrids - v.y - 1); dy++) {
                        let dxMin = max(-1, -v.x);
                        let dxMax = min(1, env.xGrids - v.x - 1);
                        let startCellNum = cellNumberFromId(v.x + dxMin, v.y + dy, v.z + dz);
                        let endCellNum = cellNumberFromId(v.x + dxMax, v.y + dy, v.z + dz);
                        let start = prefixSum[startCellNum];
                        let end = prefixSum[endCellNum + 1];
                        for (var j = start; j < end; j++) {
                            let density_j = sortedParticles[j].density;
                            let nearDensity_j = sortedParticles[j].nearDensity;
                            let pos_j = sortedParticles[j].position;
                            let r2 = dot(pos_i - pos_j, pos_i - pos_j); 
                            if (density_j == 0. || nearDensity_j == 0.) {
                                continue;
                            }
                            if (r2 < params.kernelRadiusPow2 && 1e-64 < r2) {
                                let r = sqrt(r2);
                                let pressure_i = params.stiffness * (density_i - params.restDensity);
                                let pressure_j = params.stiffness * (density_j - params.restDensity);
                                let nearPressure_i = params.nearStiffness * nearDensity_i;
                                let nearPressure_j = params.nearStiffness * nearDensity_j;
                                let sharedPressure = (pressure_i + pressure_j) / 2.0;
                                let nearSharedPressure = (nearPressure_i + nearPressure_j) / 2.0;
                                let dir = normalize(pos_j - pos_i);
                                fPress += -params.mass * sharedPressure * dir * densityKernelGradient(r) / density_j;
                                fPress += -params.mass * nearSharedPressure * dir * nearDensityKernelGradient(r) / nearDensity_j;
                                let relativeSpeed = sortedParticles[j].v - particles[id.x].v;
                                fVisc += params.mass * relativeSpeed * viscosityKernelLaplacian(r) / density_j;
                            }
                        }
                    }
                }
            }
        }

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        fVisc *= params.viscosity;
        let fGrv: vec3f = density_i * vec3f(0.0, -9.8, 0.0);
        particles[id.x].force = fPress + fVisc + fGrv;
    }
}`,Un=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct RealBoxSize {
  xHalf: f32, 
  yHalf: f32, 
  zHalf: f32, 
}

struct SPHParams {
    mass: f32, 
    kernelRadius: f32, 
    kernelRadiusPow2: f32, 
    kernelRadiusPow5: f32, 
    kernelRadiusPow6: f32,  
    kernelRadiusPow9: f32, 
    dt: f32, 
    stiffness: f32, 
    nearStiffness: f32, 
    restDensity: f32, 
    viscosity: f32, 
    n: u32
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> realBoxSize: RealBoxSize;
@group(0) @binding(2) var<uniform> params: SPHParams;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.n) {
    
    if (particles[id.x].density != 0.) {
      var a = particles[id.x].force / particles[id.x].density;

      let xPlusDist = realBoxSize.xHalf - particles[id.x].position.x;
      let xMinusDist = realBoxSize.xHalf + particles[id.x].position.x;
      let yPlusDist = realBoxSize.yHalf - particles[id.x].position.y;
      let yMinusDist = realBoxSize.yHalf + particles[id.x].position.y;
      let zPlusDist = realBoxSize.zHalf - particles[id.x].position.z;
      let zMinusDist = realBoxSize.zHalf + particles[id.x].position.z;

      let wallStiffness = 8000.;

      let xPlusForce = vec3f(1., 0., 0.) * wallStiffness * min(xPlusDist, 0.);
      let xMinusForce = vec3f(-1., 0., 0.) * wallStiffness * min(xMinusDist, 0.);
      let yPlusForce = vec3f(0., 1., 0.) * wallStiffness * min(yPlusDist, 0.);
      let yMinusForce = vec3f(0., -1., 0.) * wallStiffness * min(yMinusDist, 0.);
      let zPlusForce = vec3f(0., 0., 1.) * wallStiffness * min(zPlusDist, 0.);
      let zMinusForce = vec3f(0., 0., -1.) * wallStiffness * min(zMinusDist, 0.);

      let xForce = xPlusForce + xMinusForce;
      let yForce = yPlusForce + yMinusForce;
      let zForce = zPlusForce + zMinusForce;

      a += xForce + yForce + zForce;
      particles[id.x].v += params.dt * a;
      particles[id.x].position += params.dt * particles[id.x].v;
    }
  }
}`,Fn=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct PosVel {
    position: vec3f, 
    v: vec3f, 
}

struct SPHParams {
    mass: f32, 
    kernelRadius: f32, 
    kernelRadiusPow2: f32, 
    kernelRadiusPow5: f32, 
    kernelRadiusPow6: f32,  
    kernelRadiusPow9: f32, 
    dt: f32, 
    stiffness: f32, 
    nearStiffness: f32, 
    restDensity: f32, 
    viscosity: f32, 
    n: u32
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> posvel: array<PosVel>;
@group(0) @binding(2) var<uniform> env: SPHParams;

@compute @workgroup_size(64)
fn copyPosition(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < env.n) {
        posvel[id.x].position = particles[id.x].position;
        posvel[id.x].v = particles[id.x].v;
    }
}`;const An=`

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    
    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID
    
    // Load input to shared memory
    temp[ELM_TID]     = select(items[ELM_GID], 0, ELM_GID >= ELEMENT_COUNT);
    temp[ELM_TID + 1] = select(items[ELM_GID + 1], 0, ELM_GID + 1 >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        let last_offset = ITEMS_PER_WORKGROUP - 1;

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (ELM_GID >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID] = temp[ELM_TID];

    if (ELM_GID + 1 >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID + 1] = temp[ELM_TID + 1];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    items[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    items[ELM_ID + 1] += blockSum;
}`,Ln=`

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

const NUM_BANKS: u32 = 32;
const LOG_NUM_BANKS: u32 = 5;

fn get_offset(offset: u32) -> u32 {
    // return offset >> LOG_NUM_BANKS; // Conflict-free
    return (offset >> NUM_BANKS) + (offset >> (2 * LOG_NUM_BANKS)); // Zero bank conflict
}

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    
    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID
    
    // Load input to shared memory
    let ai: u32 = TID;
    let bi: u32 = TID + (ITEMS_PER_WORKGROUP >> 1);
    let s_ai = ai + get_offset(ai);
    let s_bi = bi + get_offset(bi);
    let g_ai = ai + WID * 2;
    let g_bi = bi + WID * 2;
    temp[s_ai] = select(items[g_ai], 0, g_ai >= ELEMENT_COUNT);
    temp[s_bi] = select(items[g_bi], 0, g_bi >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            ai += get_offset(ai);
            bi += get_offset(bi);
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        var last_offset = ITEMS_PER_WORKGROUP - 1;
        last_offset += get_offset(last_offset);

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            ai += get_offset(ai);
            bi += get_offset(bi);

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (g_ai < ELEMENT_COUNT) {
        items[g_ai] = temp[s_ai];
    }
    if (g_bi < ELEMENT_COUNT) {
        items[g_bi] = temp[s_bi];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    items[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    items[ELM_ID + 1] += blockSum;
}`;function Vn(f,h){const v={x:h,y:1};if(h>f.limits.maxComputeWorkgroupsPerDimension){const z=Math.floor(Math.sqrt(h)),M=Math.ceil(h/z);v.x=z,v.y=M}return v}class Nn{constructor({device:h,data:v,count:z,workgroup_size:M={x:16,y:16},avoid_bank_conflicts:G=!1}){if(this.device=h,this.workgroup_size=M,this.threads_per_workgroup=M.x*M.y,this.items_per_workgroup=2*this.threads_per_workgroup,Math.log2(this.threads_per_workgroup)%1!==0)throw new Error(`workgroup_size.x * workgroup_size.y must be a power of two. (current: ${this.threads_per_workgroup})`);this.pipelines=[],this.shaderModule=this.device.createShaderModule({label:"prefix-sum",code:G?Ln:An}),this.create_pass_recursive(v,z)}create_pass_recursive(h,v){const z=Math.ceil(v/this.items_per_workgroup),M=Vn(this.device,z),G=this.device.createBuffer({label:"prefix-sum-block-sum",size:z*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),I=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),E=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:I,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:G}}]}),F=this.device.createPipelineLayout({bindGroupLayouts:[I]}),K=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:F,compute:{module:this.shaderModule,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ITEMS_PER_WORKGROUP:this.items_per_workgroup,ELEMENT_COUNT:v}}});if(this.pipelines.push({pipeline:K,bindGroup:E,dispatchSize:M}),z>1){this.create_pass_recursive(G,z);const Z=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:F,compute:{module:this.shaderModule,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ELEMENT_COUNT:v}}});this.pipelines.push({pipeline:Z,bindGroup:E,dispatchSize:M})}}get_dispatch_chain(){return this.pipelines.flatMap(h=>[h.dispatchSize.x,h.dispatchSize.y,1])}dispatch(h,v,z=0){for(let M=0;M<this.pipelines.length;M++){const{pipeline:G,bindGroup:I,dispatchSize:E}=this.pipelines[M];h.setPipeline(G),h.setBindGroup(0,I),v==null?h.dispatchWorkgroups(E.x,E.y,1):h.dispatchWorkgroupsIndirect(v,z+M*3*4)}}}const je=64;class Wn{constructor(h,v,z,M){b(this,"device");b(this,"gridClearPipeline");b(this,"gridBuildPipeline");b(this,"reorderPipeline");b(this,"densityPipeline");b(this,"forcePipeline");b(this,"integratePipeline");b(this,"copyPositionPipeline");b(this,"gridClearBindGroup");b(this,"gridBuildBindGroup");b(this,"reorderBindGroup");b(this,"densityBindGroup");b(this,"forceBindGroup");b(this,"integrateBindGroup");b(this,"copyPositionBindGroup");b(this,"cellParticleCountBuffer");b(this,"particleBuffer");b(this,"realBoxSizeBuffer");b(this,"sphParamsBuffer");b(this,"prefixSumKernel");b(this,"kernelRadius",.07);b(this,"numParticles",0);b(this,"gridCount",0);b(this,"renderDiameter");this.device=M,this.renderDiameter=z;const G=M.createShaderModule({code:On}),I=M.createShaderModule({code:kn}),E=M.createShaderModule({code:Un}),F=M.createShaderModule({code:In}),K=M.createShaderModule({code:Tn}),Z=M.createShaderModule({code:En}),U=M.createShaderModule({code:Fn}),N=1*this.kernelRadius,q=2,Q=2,X=2,Y=2*q,ee=2*Q,se=2*X,ie=4*N,le=Math.ceil((Y+ie)/N),te=Math.ceil((ee+ie)/N),oe=Math.ceil((se+ie)/N);this.gridCount=le*te*oe;const ue=ie/2,fe=20,de=1,ne=1,xe=15e3,k=100,V=.006;this.gridClearPipeline=M.createComputePipeline({label:"grid clear pipeline",layout:"auto",compute:{module:K}}),this.gridBuildPipeline=M.createComputePipeline({label:"grid build pipeline",layout:"auto",compute:{module:F}}),this.reorderPipeline=M.createComputePipeline({label:"reorder pipeline",layout:"auto",compute:{module:Z}}),this.densityPipeline=M.createComputePipeline({label:"density pipeline",layout:"auto",compute:{module:G}}),this.forcePipeline=M.createComputePipeline({label:"force pipeline",layout:"auto",compute:{module:I}}),this.integratePipeline=M.createComputePipeline({label:"integrate pipeline",layout:"auto",compute:{module:E}}),this.copyPositionPipeline=M.createComputePipeline({label:"copy position pipeline",layout:"auto",compute:{module:U}});const T=new ArrayBuffer(32),J={xGrids:new Int32Array(T,0,1),yGrids:new Int32Array(T,4,1),zGrids:new Int32Array(T,8,1),cellSize:new Float32Array(T,12,1),xHalf:new Float32Array(T,16,1),yHalf:new Float32Array(T,20,1),zHalf:new Float32Array(T,24,1),offset:new Float32Array(T,28,1)};J.xGrids.set([le]),J.yGrids.set([te]),J.zGrids.set([oe]),J.cellSize.set([N]),J.xHalf.set([q]),J.yHalf.set([Q]),J.zHalf.set([X]),J.offset.set([ue]);const W=new ArrayBuffer(48),$={mass:new Float32Array(W,0,1),kernelRadius:new Float32Array(W,4,1),kernelRadiusPow2:new Float32Array(W,8,1),kernelRadiusPow5:new Float32Array(W,12,1),kernelRadiusPow6:new Float32Array(W,16,1),kernelRadiusPow9:new Float32Array(W,20,1),dt:new Float32Array(W,24,1),stiffness:new Float32Array(W,28,1),nearStiffness:new Float32Array(W,32,1),restDensity:new Float32Array(W,36,1),viscosity:new Float32Array(W,40,1),n:new Uint32Array(W,44,1)};$.mass.set([ne]),$.kernelRadius.set([this.kernelRadius]),$.kernelRadiusPow2.set([Math.pow(this.kernelRadius,2)]),$.kernelRadiusPow5.set([Math.pow(this.kernelRadius,5)]),$.kernelRadiusPow6.set([Math.pow(this.kernelRadius,6)]),$.kernelRadiusPow9.set([Math.pow(this.kernelRadius,9)]),$.dt.set([V]),$.stiffness.set([fe]),$.nearStiffness.set([de]),$.restDensity.set([xe]),$.viscosity.set([k]);const ce=new ArrayBuffer(12);this.cellParticleCountBuffer=M.createBuffer({label:"cell particle count buffer",size:4*(this.gridCount+1),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const ge=M.createBuffer({label:"target particles buffer",size:je*Ae,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),j=M.createBuffer({label:"particle cell offset buffer",size:4*Ae,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.realBoxSizeBuffer=M.createBuffer({label:"real box size buffer",size:ce.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const ae=M.createBuffer({label:"environment buffer",size:T.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.sphParamsBuffer=M.createBuffer({label:"sph params buffer",size:W.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),M.queue.writeBuffer(ae,0,T),M.queue.writeBuffer(this.sphParamsBuffer,0,W),this.gridClearBindGroup=M.createBindGroup({layout:this.gridClearPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.cellParticleCountBuffer}}]}),this.gridBuildBindGroup=M.createBindGroup({layout:this.gridBuildPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.cellParticleCountBuffer}},{binding:1,resource:{buffer:j}},{binding:2,resource:{buffer:h}},{binding:3,resource:{buffer:ae}},{binding:4,resource:{buffer:this.sphParamsBuffer}}]}),this.reorderBindGroup=M.createBindGroup({layout:this.reorderPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:ge}},{binding:2,resource:{buffer:this.cellParticleCountBuffer}},{binding:3,resource:{buffer:j}},{binding:4,resource:{buffer:ae}},{binding:5,resource:{buffer:this.sphParamsBuffer}}]}),this.densityBindGroup=M.createBindGroup({layout:this.densityPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:ge}},{binding:2,resource:{buffer:this.cellParticleCountBuffer}},{binding:3,resource:{buffer:ae}},{binding:4,resource:{buffer:this.sphParamsBuffer}}]}),this.forceBindGroup=M.createBindGroup({layout:this.forcePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:ge}},{binding:2,resource:{buffer:this.cellParticleCountBuffer}},{binding:3,resource:{buffer:ae}},{binding:4,resource:{buffer:this.sphParamsBuffer}}]}),this.integrateBindGroup=M.createBindGroup({layout:this.integratePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:this.realBoxSizeBuffer}},{binding:2,resource:{buffer:this.sphParamsBuffer}}]}),this.copyPositionBindGroup=M.createBindGroup({layout:this.copyPositionPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:v}},{binding:2,resource:{buffer:this.sphParamsBuffer}}]}),this.particleBuffer=h}reset(h,v){Fe.sphere_size.set([this.renderDiameter]);const z=this.initDambreak(v,h),M=new ArrayBuffer(12),G={xHalf:new Float32Array(M,0,1),yHalf:new Float32Array(M,4,1),zHalf:new Float32Array(M,8,1)};G.xHalf.set([v[0]]),G.yHalf.set([v[1]]),G.zHalf.set([v[2]]);const I=new Float32Array(1);I[0]=this.numParticles,console.log(this.numParticles),this.device.queue.writeBuffer(this.sphParamsBuffer,44,I),this.device.queue.writeBuffer(this.particleBuffer,0,z),this.device.queue.writeBuffer(this.realBoxSizeBuffer,0,M)}execute(h){const v=h.beginComputePass();for(let z=0;z<2;z++)v.setBindGroup(0,this.gridClearBindGroup),v.setPipeline(this.gridClearPipeline),v.dispatchWorkgroups(Math.ceil((this.gridCount+1)/64)),v.setBindGroup(0,this.gridBuildBindGroup),v.setPipeline(this.gridBuildPipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),this.prefixSumKernel=new Nn({device:this.device,data:this.cellParticleCountBuffer,count:this.gridCount+1}),this.prefixSumKernel.dispatch(v),v.setBindGroup(0,this.reorderBindGroup),v.setPipeline(this.reorderPipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.densityBindGroup),v.setPipeline(this.densityPipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.reorderBindGroup),v.setPipeline(this.reorderPipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.forceBindGroup),v.setPipeline(this.forcePipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.integrateBindGroup),v.setPipeline(this.integratePipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64)),v.setBindGroup(0,this.copyPositionBindGroup),v.setPipeline(this.copyPositionPipeline),v.dispatchWorkgroups(Math.ceil(this.numParticles/64));v.end()}initDambreak(h,v){let z=new ArrayBuffer(je*v);this.numParticles=0;const M=.5;for(var G=-h[1]*.95;this.numParticles<v;G+=M*this.kernelRadius)for(var I=-.95*h[0];I<.95*h[0]&&this.numParticles<v;I+=M*this.kernelRadius)for(var E=-.95*h[2];E<0*h[2]&&this.numParticles<v;E+=M*this.kernelRadius){let F=.001*Math.random();const K=je*this.numParticles;({position:new Float32Array(z,K+0,3),v:new Float32Array(z,K+16,3),force:new Float32Array(z,K+32,3),density:new Float32Array(z,K+44,1),nearDensity:new Float32Array(z,K+48,1)}).position.set([I+F,G+F,E+F]),this.numParticles++}return console.log(this.numParticles),z}changeBoxSize(h){const v=new ArrayBuffer(12),z=new Float32Array(v);z.set(h),this.device.queue.writeBuffer(this.realBoxSizeBuffer,0,z)}}var qn=`struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
}

struct FragmentOutput {
    @location(0) frag_color: vec4f, 
    @builtin(frag_depth) frag_depth: f32, 
}

struct RenderUniforms {
    texel_size: vec2f, 
    sphere_size: f32, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}

struct PosVel {
    position: vec3f, 
    v: vec3f, 
}

@group(0) @binding(0) var<storage> particles: array<PosVel>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

@vertex
fn vs(    
    @builtin(vertex_index) vertex_index: u32, 
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var corner_positions = array(
        vec2( 0.5,  0.5),
        vec2( 0.5, -0.5),
        vec2(-0.5, -0.5),
        vec2( 0.5,  0.5),
        vec2(-0.5, -0.5),
        vec2(-0.5,  0.5),
    );

    
    
    let corner = vec3(corner_positions[vertex_index] * uniforms.sphere_size, 0.0);
    let uv = corner_positions[vertex_index] + 0.5;

    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;

    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);

    return VertexOutput(out_position, uv, view_position);
}

@fragment
fn fs(input: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    var normalxy: vec2f = input.uv * 2.0 - 1.0;
    var r2: f32 = dot(normalxy, normalxy);
    if (r2 > 1.0) {
        discard;
    }
    var normalz = sqrt(1.0 - r2);
    var normal = vec3(normalxy, normalz);

    var radius = uniforms.sphere_size / 2;
    var real_view_pos: vec4f = vec4f(input.view_position + normal * radius, 1.0);
    var clip_space_pos: vec4f = uniforms.projection_matrix * real_view_pos;
    out.frag_depth = clip_space_pos.z / clip_space_pos.w;

    out.frag_color = vec4(real_view_pos.z, 0., 0., 1.);
    return out;
}`,Hn=`@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

struct FragmentInput {
    @location(0) uv: vec2f,  
    @location(1) iuv: vec2f
}

override depth_threshold: f32;  
override projected_particle_constant: f32; 
override max_filter_size: f32;
struct FilterUniforms {
    blur_dir: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);

    
    if (depth >= 1e4 || depth <= 0.) {
        return vec4f(vec3f(depth), 1.);
    }

    
    var filter_size: i32 = min(i32(max_filter_size), i32(ceil(projected_particle_constant / depth)));

    
    var sigma: f32 = f32(filter_size) / 3.0; 
    var two_sigma: f32 = 2.0 * sigma * sigma;
    var sigma_depth: f32 = depth_threshold / 3.0;
    var two_sigma_depth: f32 = 2.0 * sigma_depth * sigma_depth;

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_depth: f32 = abs(textureLoad(texture, vec2u(input.iuv + coords * uniforms.blur_dir), 0).r);
        

        var rr: f32 = dot(coords, coords);
        var w: f32 = exp(-rr / two_sigma);

        var r_depth: f32 = sampled_depth - depth;
        var wd: f32 = exp(-r_depth * r_depth / two_sigma_depth);
        sum += sampled_depth * w * wd;
        wsum += w * wd;
    }

    
    
    
    

    
    

    
    

    
    
    
    

    sum /= wsum;
    
    
    

    return vec4f(sum, 0., 0., 1.);
}`,Kn=`@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(3) var thickness_texture: texture_2d<f32>;
@group(0) @binding(4) var envmap_texture: texture_cube<f32>;

struct RenderUniforms {
    texel_size: vec2f, 
    sphere_size: f32, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) iuv: vec2f, 
}

fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    var ndc: vec4f = vec4f(tex_coord.x * 2.0 - 1.0, 1.0 - 2.0 * tex_coord.y, 0.0, 1.0);
    
    ndc.z = -uniforms.projection_matrix[2].z + uniforms.projection_matrix[3].z / depth;
    ndc.w = 1.0;

    var eye_pos: vec4f = uniforms.inv_projection_matrix * ndc;

    return eye_pos.xyz / eye_pos.w;
}

fn getViewPosFromTexCoord(tex_coord: vec2f, iuv: vec2f) -> vec3f {
    var depth: f32 = abs(textureLoad(texture, vec2u(iuv), 0).x);
    return computeViewPosFromUVDepth(tex_coord, depth);
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);

    let bgColor: vec3f = vec3f(0.8, 0.8, 0.8);

    if (depth >= 1e4 || depth <= 0.) {
        return vec4f(bgColor, 1.);
    }

    var viewPos: vec3f = computeViewPosFromUVDepth(input.uv, depth); 

    var ddx: vec3f = getViewPosFromTexCoord(input.uv + vec2f(uniforms.texel_size.x, 0.), input.iuv + vec2f(1.0, 0.0)) - viewPos; 
    var ddy: vec3f = getViewPosFromTexCoord(input.uv + vec2f(0., uniforms.texel_size.y), input.iuv + vec2f(0.0, 1.0)) - viewPos; 
    var ddx2: vec3f = viewPos - getViewPosFromTexCoord(input.uv + vec2f(-uniforms.texel_size.x, 0.), input.iuv + vec2f(-1.0, 0.0));
    var ddy2: vec3f = viewPos - getViewPosFromTexCoord(input.uv + vec2f(0., -uniforms.texel_size.y), input.iuv + vec2f(0.0, -1.0));

    if (abs(ddx.z) > abs(ddx2.z)) {
        ddx = ddx2; 
    }
    if (abs(ddy.z) > abs(ddy2.z)) {
        ddy = ddy2;
    }

    var normal: vec3f = -normalize(cross(ddx, ddy)); 
    var rayDir = normalize(viewPos);
    var lightDir = normalize((uniforms.view_matrix * vec4f(0, 0, -1, 0.)).xyz);
    var H: vec3f        = normalize(lightDir - rayDir);
    var specular: f32   = pow(max(0.0, dot(H, normal)), 250.);
    var diffuse: f32  = max(0.0, dot(lightDir, normal)) * 1.0;

    var density = 1.5; 
    
    var thickness = textureLoad(thickness_texture, vec2u(input.iuv), 0).r;
    var diffuseColor = vec3f(0.085, 0.6375, 0.9);
    var transmittance: vec3f = exp(-density * thickness * (1.0 - diffuseColor)); 
    var refractionColor: vec3f = bgColor * transmittance;

    let F0 = 0.02;
    var fresnel: f32 = clamp(F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0), 0., 1.0);

    var reflectionDir: vec3f = reflect(rayDir, normal);
    var reflectionDirWorld: vec3f = (uniforms.inv_view_matrix * vec4f(reflectionDir, 0.0)).xyz;
    var reflectionColor: vec3f = textureSampleLevel(envmap_texture, texture_sampler, reflectionDirWorld, 0.).rgb; 
    var finalColor = 1.0 * specular + mix(refractionColor, reflectionColor, fresnel);

    return vec4f(finalColor, 1.0);

    

    
    
    
    
    
    
    
    
    
    
    
}`,Yn=`struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
  @location(1) iuv : vec2f,
}

override screenWidth: f32;
override screenHeight: f32;

@vertex
fn vs(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var out: VertexOutput;

    var pos = array(
        vec2( 1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
    );

    var uv = array(
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.0, 1.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(0.0, 0.0),
    );

    out.position = vec4(pos[vertex_index], 0.0, 1.0);
    out.uv = uv[vertex_index];
    out.iuv = out.uv * vec2f(screenWidth, screenHeight);

    return out;
}`,jn=`struct RenderUniforms {
    texel_size: vec2f, 
    sphere_size: f32, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}

struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
}

struct PosVel {
    position: vec3f, 
    v: vec3f, 
}

@group(0) @binding(0) var<storage> particles: array<PosVel>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

@vertex
fn vs(    
    @builtin(vertex_index) vertex_index: u32, 
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var corner_positions = array(
        vec2( 0.5,  0.5),
        vec2( 0.5, -0.5),
        vec2(-0.5, -0.5),
        vec2( 0.5,  0.5),
        vec2(-0.5, -0.5),
        vec2(-0.5,  0.5),
    );

    let corner = vec3(corner_positions[vertex_index] * uniforms.sphere_size, 0.0);
    let uv = corner_positions[vertex_index] + 0.5;

    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;

    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);

    return VertexOutput(out_position, uv);
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var normalxy: vec2f = input.uv * 2.0 - 1.0;
    var r2: f32 = dot(normalxy, normalxy);
    if (r2 > 1.0) {
        discard;
    }
    var thickness: f32 = sqrt(1.0 - r2);
    let particle_alpha = 0.05;

    return vec4f(vec3f(particle_alpha * thickness), 1.0);
}`,Xn=`@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

struct FragmentInput {
    @location(0) uv: vec2f,  
    @location(1) iuv: vec2f
}

struct FilterUniforms {
    blur_dir: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    
    var thickness: f32 = textureLoad(texture, vec2u(input.iuv), 0).r;
    if (thickness == 0.) {
        return vec4f(0., 0., 0., 1.);
    }

    
    var filter_size: i32 = 30; 
    var sigma: f32 = f32(filter_size) / 3.0;
    var two_sigma: f32 = 2.0 * sigma * sigma;

    var sum = 0.;
    var wsum = 0.;

    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_thickness: f32 = textureLoad(texture, vec2u(input.iuv + uniforms.blur_dir * coords), 0).r;

        var w: f32 = exp(-coords.x * coords.x / two_sigma);

        sum += sampled_thickness * w;
        wsum += w;
    }

    sum /= wsum;

    return vec4f(sum, 0., 0., 1.);
}`,Zn=`struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
    @location(2) speed: f32, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
    @location(2) speed: f32, 
}

struct FragmentOutput {
    @location(0) frag_color: vec4f, 
    @builtin(frag_depth) frag_depth: f32, 
}

struct RenderUniforms {
    texel_size: vec2f, 
    sphere_size: f32, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}

struct PosVel {
    position: vec3f, 
    v: vec3f, 
}

@group(0) @binding(0) var<storage> particles: array<PosVel>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

@vertex
fn vs(    
    @builtin(vertex_index) vertex_index: u32, 
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var corner_positions = array(
        vec2( 0.5,  0.5),
        vec2( 0.5, -0.5),
        vec2(-0.5, -0.5),
        vec2( 0.5,  0.5),
        vec2(-0.5, -0.5),
        vec2(-0.5,  0.5),
    );

    let corner = vec3(corner_positions[vertex_index] * uniforms.sphere_size, 0.0);
    let uv = corner_positions[vertex_index] + 0.5;

    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;

    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);

    let speed = sqrt(dot(particles[instance_index].v, particles[instance_index].v));

    return VertexOutput(out_position, uv, view_position, speed);
}

fn value_to_color(value: f32) -> vec3<f32> {
    
    let col0 = vec3f(0, 0.4, 0.8);
    let col1 = vec3f(35, 161, 165) / 256;
    let col2 = vec3f(95, 254, 150) / 256;
    let col3 = vec3f(243, 250, 49) / 256;
    let col4 = vec3f(255, 165, 0) / 256;

    if (0 <= value && value < 0.25) {
        let t = value / 0.25;
        return mix(col0, col1, t);
    } else if (0.25 <= value && value < 0.50) {
        let t = (value - 0.25) / 0.25;
        return mix(col1, col2, t);
    } else if (0.50 <= value && value < 0.75) {
        let t = (value - 0.50) / 0.25;
        return mix(col2, col3, t);
    } else {
        let t = (value - 0.75) / 0.25;
        return mix(col3, col4, t);
    }

}

@fragment
fn fs(input: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    var normalxy: vec2f = input.uv * 2.0 - 1.0;
    var r2: f32 = dot(normalxy, normalxy);
    if (r2 > 1.0) {
        discard;
    }
    var normalz = sqrt(1.0 - r2);
    var normal = vec3(normalxy, normalz);

    var radius = uniforms.sphere_size / 2;
    var real_view_pos: vec4f = vec4f(input.view_position + normal * radius, 1.0);
    var clip_space_pos: vec4f = uniforms.projection_matrix * real_view_pos;
    out.frag_depth = clip_space_pos.z / clip_space_pos.w;

    var diffuse: f32 = max(0.0, dot(normal, normalize(vec3(1.0, 1.0, 1.0))));
    var color: vec3f = value_to_color(input.speed / 1.5);

    out.frag_color = vec4(color * diffuse, 1.);
    return out;
}`;class cn{constructor(h,v,z,M,G,I,E,F){b(this,"depthMapPipeline");b(this,"depthFilterPipeline");b(this,"thicknessMapPipeline");b(this,"thicknessFilterPipeline");b(this,"fluidPipeline");b(this,"spherePipeline");b(this,"depthMapTextureView");b(this,"tmpDepthMapTextureView");b(this,"thicknessTextureView");b(this,"tmpThicknessTextureView");b(this,"depthTestTextureView");b(this,"depthMapBindGroup");b(this,"depthFilterBindGroups");b(this,"thicknessMapBindGroup");b(this,"thicknessFilterBindGroups");b(this,"fluidBindGroup");b(this,"sphereBindGroup");b(this,"device");this.device=h;const K=100,Z=10,U=2*M,N=12,q={screenHeight:v.height,screenWidth:v.width},Q={depth_threshold:M*Z,max_filter_size:K,projected_particle_constant:N*U*.05*(v.height/2)/Math.tan(G/2)},X=h.createSampler({magFilter:"linear",minFilter:"linear"}),Y=h.createShaderModule({code:Yn}),ee=h.createShaderModule({code:qn}),se=h.createShaderModule({code:Hn}),ie=h.createShaderModule({code:Kn}),le=h.createShaderModule({code:Zn}),te=h.createShaderModule({code:jn}),oe=h.createShaderModule({code:Xn});this.spherePipeline=h.createRenderPipeline({label:"ball pipeline",layout:"auto",vertex:{module:le},fragment:{module:le,targets:[{format:z}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),this.depthMapPipeline=h.createRenderPipeline({label:"depth map pipeline",layout:"auto",vertex:{module:ee},fragment:{module:ee,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),this.depthFilterPipeline=h.createRenderPipeline({label:"filter pipeline",layout:"auto",vertex:{module:Y,constants:q},fragment:{module:se,constants:Q,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"}}),this.thicknessMapPipeline=h.createRenderPipeline({label:"thickness map pipeline",layout:"auto",vertex:{module:te},fragment:{module:te,targets:[{format:"r16float",writeMask:GPUColorWrite.RED,blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"}}),this.thicknessFilterPipeline=h.createRenderPipeline({label:"thickness filter pipeline",layout:"auto",vertex:{module:Y,constants:q},fragment:{module:oe,targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),this.fluidPipeline=h.createRenderPipeline({label:"fluid rendering pipeline",layout:"auto",vertex:{module:Y,constants:q},fragment:{module:ie,targets:[{format:z}]},primitive:{topology:"triangle-list"}});const ue=h.createTexture({label:"depth map texture",size:[v.width,v.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}),fe=h.createTexture({label:"temporary depth map texture",size:[v.width,v.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}),de=h.createTexture({label:"thickness map texture",size:[v.width,v.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}),ne=h.createTexture({label:"temporary thickness map texture",size:[v.width,v.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}),xe=h.createTexture({size:[v.width,v.height,1],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT});this.depthMapTextureView=ue.createView(),this.tmpDepthMapTextureView=fe.createView(),this.thicknessTextureView=de.createView(),this.tmpThicknessTextureView=ne.createView(),this.depthTestTextureView=xe.createView();const k=new ArrayBuffer(8),V=new ArrayBuffer(8),T={blur_dir:new Float32Array(k)},J={blur_dir:new Float32Array(V)};T.blur_dir.set([1,0]),J.blur_dir.set([0,1]);const W=h.createBuffer({label:"filter uniform buffer",size:k.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),$=h.createBuffer({label:"filter uniform buffer",size:V.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});h.queue.writeBuffer(W,0,k),h.queue.writeBuffer($,0,V),this.depthMapBindGroup=h.createBindGroup({label:"depth map bind group",layout:this.depthMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:I}},{binding:1,resource:{buffer:E}}]}),this.depthFilterBindGroups=[],this.depthFilterBindGroups=[h.createBindGroup({label:"filterX bind group",layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:W}}]}),h.createBindGroup({label:"filterY bind group",layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpDepthMapTextureView},{binding:2,resource:{buffer:$}}]})],this.thicknessMapBindGroup=h.createBindGroup({label:"thickness map bind group",layout:this.thicknessMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:I}},{binding:1,resource:{buffer:E}}]}),this.thicknessFilterBindGroups=[],this.thicknessFilterBindGroups=[h.createBindGroup({label:"thickness filterX bind group",layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.thicknessTextureView},{binding:2,resource:{buffer:W}}]}),h.createBindGroup({label:"thickness filterY bind group",layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpThicknessTextureView},{binding:2,resource:{buffer:$}}]})],this.fluidBindGroup=h.createBindGroup({label:"fluid bind group",layout:this.fluidPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:X},{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:E}},{binding:3,resource:this.thicknessTextureView},{binding:4,resource:F}]}),this.sphereBindGroup=h.createBindGroup({label:"ball bind group",layout:this.spherePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:I}},{binding:1,resource:{buffer:E}}]})}execute(h,v,z,M){const G={colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTestTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}},I=[{colorAttachments:[{view:this.tmpDepthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}],E={colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},F=[{colorAttachments:[{view:this.tmpThicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}],K={colorAttachments:[{view:h.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Z={colorAttachments:[{view:h.getCurrentTexture().createView(),clearValue:{r:.8,g:.8,b:.8,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTestTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}};if(M){const N=v.beginRenderPass(Z);N.setBindGroup(0,this.sphereBindGroup),N.setPipeline(this.spherePipeline),N.draw(6,z),N.end()}else{const N=v.beginRenderPass(G);N.setBindGroup(0,this.depthMapBindGroup),N.setPipeline(this.depthMapPipeline),N.draw(6,z),N.end();for(var U=0;U<4;U++){const X=v.beginRenderPass(I[0]);X.setBindGroup(0,this.depthFilterBindGroups[0]),X.setPipeline(this.depthFilterPipeline),X.draw(6),X.end();const Y=v.beginRenderPass(I[1]);Y.setBindGroup(0,this.depthFilterBindGroups[1]),Y.setPipeline(this.depthFilterPipeline),Y.draw(6),Y.end()}const q=v.beginRenderPass(E);q.setBindGroup(0,this.thicknessMapBindGroup),q.setPipeline(this.thicknessMapPipeline),q.draw(6,z),q.end();for(var U=0;U<1;U++){const Y=v.beginRenderPass(F[0]);Y.setBindGroup(0,this.thicknessFilterBindGroups[0]),Y.setPipeline(this.thicknessFilterPipeline),Y.draw(6),Y.end();const ee=v.beginRenderPass(F[1]);ee.setBindGroup(0,this.thicknessFilterBindGroups[1]),ee.setPipeline(this.thicknessFilterPipeline),ee.draw(6),ee.end()}const Q=v.beginRenderPass(K);Q.setBindGroup(0,this.fluidBindGroup),Q.setPipeline(this.fluidPipeline),Q.draw(6),Q.end()}}}async function $n(){const f=document.querySelector("canvas");if(!navigator.gpu)throw alert("WebGPU is not supported on your browser."),new Error;const h=await navigator.gpu.requestAdapter();if(!h)throw alert("Adapter is not available."),new Error;const v=await h.requestDevice(),z=f.getContext("webgpu");if(!z)throw new Error;let M=.7;f.width=M*f.clientWidth,f.height=M*f.clientHeight;const G=navigator.gpu.getPreferredCanvasFormat();return z.configure({device:v,format:G}),{canvas:f,device:v,presentationFormat:G,context:z}}async function Qn(){const{canvas:f,device:h,presentationFormat:v,context:z}=await $n();console.log("initialization done"),z.configure({device:h,format:v});let M;{const w=["cubemap/posx.png","cubemap/negx.png","cubemap/posy.png","cubemap/negy.png","cubemap/posz.png","cubemap/negz.png"].map(async o=>{const n=await fetch(o);return createImageBitmap(await n.blob())}),e=await Promise.all(w);M=h.createTexture({dimension:"2d",size:[e[0].width,e[0].height,6],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});for(let o=0;o<e.length;o++){const n=e[o];h.queue.copyExternalImageToTexture({source:n},{texture:M,origin:[0,0,o]},[n.width,n.height])}}const G=M.createView({dimension:"cube"});console.log("cubemap initialization done"),Fe.texel_size.set([1/f.width,1/f.height]);const I=Math.max(Ye,je),E=h.createBuffer({label:"particles buffer",size:I*Ae,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),F=h.createBuffer({label:"position buffer",size:32*Ae,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),K=h.createBuffer({label:"filter uniform buffer",size:Ue.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});console.log("buffer allocating done");let Z=[4e4,7e4,12e4,2e5],U=[[35,25,55],[40,30,60],[45,40,80],[50,50,80]],N=[60,70,90,100],q=[1e4,2e4,3e4,4e4],Q=[[.7,2,.7],[1,2,1],[1.2,2,1.2],[1.4,2,1.4]],X=[2.6,3,3.4,3.8];const Y=document.getElementById("fluidCanvas"),ee=45*Math.PI/180,se=.6,ie=2*se,le=1.5,te=new Rn(E,F,ie,h),oe=45*Math.PI/180,ue=.04,fe=2*ue,de=.05,ne=new Wn(E,F,fe,h),xe=new cn(h,f,v,se,ee,F,K,G),k=new cn(h,f,v,ue,oe,F,K,G);console.log("simulator initialization done");const V=new Mn(Y);let T=document.getElementById("number-button"),J=!1,W="1";T.addEventListener("change",function(d){const w=d.target;(w==null?void 0:w.name)==="options"&&(J=!0,W=w.value)});let $=document.getElementById("simulation-mode"),ce=!1,ge="mls-mpm";$.addEventListener("change",function(d){const w=d.target;(w==null?void 0:w.name)==="options"&&(ce=!0,ge=w.value)});const j=document.getElementById("small-value"),ae=document.getElementById("medium-value"),_e=document.getElementById("large-value"),be=document.getElementById("very-large-value");let ve=document.getElementById("error-reason");ve.textContent="",h.lost.then(d=>{const w=d.reason?`reason: ${d.reason}`:"unknown reason";ve.textContent=w});const Te=N[1];let C=U[1],r=[...C];te.reset(Z[1],U[1]),V.reset(Y,Te,[C[0]/2,C[1]/4,C[2]/2],ee,le),j.textContent="40,000",ae.textContent="70,000",_e.textContent="120,000",be.textContent="200,000";let p=!1,t=!1,s=1;console.log("simulation start");async function u(){if(performance.now(),ce&&(ge=="mlsmpm"?(t=!1,j.textContent="40,000",ae.textContent="70,000",_e.textContent="120,000",be.textContent="200,000"):(t=!0,j.textContent="10,000",ae.textContent="20,000",_e.textContent="30,000",be.textContent="40,000"),ce=!1,J=!0),J){const a=parseInt(W);t?(C=Q[a],ne.reset(q[a],C),V.reset(Y,X[a],[0,-C[1]+.1,0],oe,de)):(C=U[a],te.reset(Z[a],C),V.reset(Y,N[a],[C[0]/2,C[1]/4,C[2]/2],ee,le)),r=[...C];let g=document.getElementById("slider");g.value="100",J=!1}const d=document.getElementById("slider");p=document.getElementById("particle").checked;let e=parseInt(d.value)/200+.5;const o=t?-.015:-.007,n=Math.max(e-s,o);s+=n,r[2]=C[2]*s,t?ne.changeBoxSize(r):te.changeBoxSize(r),h.queue.writeBuffer(K,0,Ue);const i=h.createCommandEncoder();t?(ne.execute(i),k.execute(z,i,ne.numParticles,p)):(te.execute(i),xe.execute(z,i,te.numParticles,p)),h.queue.submit([i.finish()]),performance.now(),requestAnimationFrame(u)}requestAnimationFrame(u)}Qn();
