import{a as le}from"./copy-shader.js";import{a as Xe,b as Ye}from"./pass.js";import{Cd as $,Dd as ee,F as Ze,Ga as Oe,H as Ve,Ha as Ne,Ka as Ie,Oc as Ke,Pc as Ee,Ra as ze,Rc as qe,Sa as Le,Vd as Be,_c as je,df as Qe,gd as ke,ld as We,na as ye,o as Z,oe as Ge,qa as be,t as Fe,u as Re,z as _e}from"./three-core.js";var ce=class{constructor(e=Math){this.grad3=[[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]],this.grad4=[[0,1,1,1],[0,1,1,-1],[0,1,-1,1],[0,1,-1,-1],[0,-1,1,1],[0,-1,1,-1],[0,-1,-1,1],[0,-1,-1,-1],[1,0,1,1],[1,0,1,-1],[1,0,-1,1],[1,0,-1,-1],[-1,0,1,1],[-1,0,1,-1],[-1,0,-1,1],[-1,0,-1,-1],[1,1,0,1],[1,1,0,-1],[1,-1,0,1],[1,-1,0,-1],[-1,1,0,1],[-1,1,0,-1],[-1,-1,0,1],[-1,-1,0,-1],[1,1,1,0],[1,1,-1,0],[1,-1,1,0],[1,-1,-1,0],[-1,1,1,0],[-1,1,-1,0],[-1,-1,1,0],[-1,-1,-1,0]],this.p=[];for(let t=0;t<256;t++)this.p[t]=Math.floor(e.random()*256);this.perm=[];for(let t=0;t<512;t++)this.perm[t]=this.p[t&255];this.simplex=[[0,1,2,3],[0,1,3,2],[0,0,0,0],[0,2,3,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,3,0],[0,2,1,3],[0,0,0,0],[0,3,1,2],[0,3,2,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,3,2,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,2,0,3],[0,0,0,0],[1,3,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,3,0,1],[2,3,1,0],[1,0,2,3],[1,0,3,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,3,1],[0,0,0,0],[2,1,3,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,1,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,0,1,2],[3,0,2,1],[0,0,0,0],[3,1,2,0],[2,1,0,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],[3,1,0,2],[0,0,0,0],[3,2,0,1],[3,2,1,0]]}noise(e,t){let i,a,r,o=.5*(Math.sqrt(3)-1),s=(e+t)*o,b=Math.floor(e+s),n=Math.floor(t+s),D=(3-Math.sqrt(3))/6,C=(b+n)*D,y=b-C,f=n-C,g=e-y,T=t-f,R,_;g>T?(R=1,_=0):(R=0,_=1);let p=g-R+D,d=T-_+D,m=g-1+2*D,M=T-1+2*D,x=b&255,S=n&255,w=this.perm[x+this.perm[S]]%12,l=this.perm[x+R+this.perm[S+_]]%12,c=this.perm[x+1+this.perm[S+1]]%12,h=.5-g*g-T*T;h<0?i=0:(h*=h,i=h*h*this._dot(this.grad3[w],g,T));let u=.5-p*p-d*d;u<0?a=0:(u*=u,a=u*u*this._dot(this.grad3[l],p,d));let P=.5-m*m-M*M;return P<0?r=0:(P*=P,r=P*P*this._dot(this.grad3[c],m,M)),70*(i+a+r)}noise3d(e,t,i){let a,r,o,s,n=(e+t+i)*.3333333333333333,D=Math.floor(e+n),C=Math.floor(t+n),y=Math.floor(i+n),f=1/6,g=(D+C+y)*f,T=D-g,R=C-g,_=y-g,p=e-T,d=t-R,m=i-_,M,x,S,w,l,c;p>=d?d>=m?(M=1,x=0,S=0,w=1,l=1,c=0):p>=m?(M=1,x=0,S=0,w=1,l=0,c=1):(M=0,x=0,S=1,w=1,l=0,c=1):d<m?(M=0,x=0,S=1,w=0,l=1,c=1):p<m?(M=0,x=1,S=0,w=0,l=1,c=1):(M=0,x=1,S=0,w=1,l=1,c=0);let h=p-M+f,u=d-x+f,P=m-S+f,O=p-w+2*f,I=d-l+2*f,z=m-c+2*f,L=p-1+3*f,K=d-1+3*f,v=m-1+3*f,A=D&255,U=C&255,F=y&255,se=this.perm[A+this.perm[U+this.perm[F]]]%12,re=this.perm[A+M+this.perm[U+x+this.perm[F+S]]]%12,oe=this.perm[A+w+this.perm[U+l+this.perm[F+c]]]%12,ne=this.perm[A+1+this.perm[U+1+this.perm[F+1]]]%12,N=.6-p*p-d*d-m*m;N<0?a=0:(N*=N,a=N*N*this._dot3(this.grad3[se],p,d,m));let E=.6-h*h-u*u-P*P;E<0?r=0:(E*=E,r=E*E*this._dot3(this.grad3[re],h,u,P));let j=.6-O*O-I*I-z*z;j<0?o=0:(j*=j,o=j*j*this._dot3(this.grad3[oe],O,I,z));let k=.6-L*L-K*K-v*v;return k<0?s=0:(k*=k,s=k*k*this._dot3(this.grad3[ne],L,K,v)),32*(a+r+o+s)}noise4d(e,t,i,a){let r=this.grad4,o=this.simplex,s=this.perm,b=(Math.sqrt(5)-1)/4,n=(5-Math.sqrt(5))/20,D,C,y,f,g,T=(e+t+i+a)*b,R=Math.floor(e+T),_=Math.floor(t+T),p=Math.floor(i+T),d=Math.floor(a+T),m=(R+_+p+d)*n,M=R-m,x=_-m,S=p-m,w=d-m,l=e-M,c=t-x,h=i-S,u=a-w,P=l>c?32:0,O=l>h?16:0,I=c>h?8:0,z=l>u?4:0,L=c>u?2:0,K=h>u?1:0,v=P+O+I+z+L+K,A=o[v][0]>=3?1:0,U=o[v][1]>=3?1:0,F=o[v][2]>=3?1:0,se=o[v][3]>=3?1:0,re=o[v][0]>=2?1:0,oe=o[v][1]>=2?1:0,ne=o[v][2]>=2?1:0,N=o[v][3]>=2?1:0,E=o[v][0]>=1?1:0,j=o[v][1]>=1?1:0,k=o[v][2]>=1?1:0,Ue=o[v][3]>=1?1:0,he=l-A+n,ue=c-U+n,me=h-F+n,pe=u-se+n,de=l-re+2*n,fe=c-oe+2*n,ve=h-ne+2*n,ge=u-N+2*n,Me=l-E+3*n,xe=c-j+3*n,Se=h-k+3*n,De=u-Ue+3*n,Te=l-1+4*n,we=c-1+4*n,Pe=h-1+4*n,Ce=u-1+4*n,q=R&255,W=_&255,B=p&255,G=d&255,He=s[q+s[W+s[B+s[G]]]]%32,Je=s[q+A+s[W+U+s[B+F+s[G+se]]]]%32,$e=s[q+re+s[W+oe+s[B+ne+s[G+N]]]]%32,et=s[q+E+s[W+j+s[B+k+s[G+Ue]]]]%32,tt=s[q+1+s[W+1+s[B+1+s[G+1]]]]%32,Q=.6-l*l-c*c-h*h-u*u;Q<0?D=0:(Q*=Q,D=Q*Q*this._dot4(r[He],l,c,h,u));let X=.6-he*he-ue*ue-me*me-pe*pe;X<0?C=0:(X*=X,C=X*X*this._dot4(r[Je],he,ue,me,pe));let Y=.6-de*de-fe*fe-ve*ve-ge*ge;Y<0?y=0:(Y*=Y,y=Y*Y*this._dot4(r[$e],de,fe,ve,ge));let H=.6-Me*Me-xe*xe-Se*Se-De*De;H<0?f=0:(H*=H,f=H*H*this._dot4(r[et],Me,xe,Se,De));let J=.6-Te*Te-we*we-Pe*Pe-Ce*Ce;return J<0?g=0:(J*=J,g=J*J*this._dot4(r[tt],Te,we,Pe,Ce)),27*(D+C+y+f+g)}_dot(e,t,i){return e[0]*t+e[1]*i}_dot3(e,t,i,a){return e[0]*t+e[1]*i+e[2]*a}_dot4(e,t,i,a,r){return e[0]*t+e[1]*i+e[2]*a+e[3]*r}};var te={name:"SSAOShader",defines:{PERSPECTIVE_CAMERA:1,KERNEL_SIZE:32},uniforms:{tNormal:{value:null},tDepth:{value:null},tNoise:{value:null},kernel:{value:null},cameraNear:{value:null},cameraFar:{value:null},resolution:{value:new Ee},cameraProjectionMatrix:{value:new ke},cameraInverseProjectionMatrix:{value:new ke},kernelRadius:{value:8},minDistance:{value:.005},maxDistance:{value:.05}},vertexShader:`

		varying vec2 vUv;

		void main() {

			vUv = uv;

			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`
		uniform highp sampler2D tNormal;
		uniform highp sampler2D tDepth;
		uniform sampler2D tNoise;

		uniform vec3 kernel[ KERNEL_SIZE ];

		uniform vec2 resolution;

		uniform float cameraNear;
		uniform float cameraFar;
		uniform mat4 cameraProjectionMatrix;
		uniform mat4 cameraInverseProjectionMatrix;

		uniform float kernelRadius;
		uniform float minDistance;
		uniform float maxDistance;

		varying vec2 vUv;

		#include <packing>

		float getDepth( const in vec2 screenPosition ) {

			return texture2D( tDepth, screenPosition ).x;

		}

		float getLinearDepth( const in vec2 screenPosition ) {

			#if PERSPECTIVE_CAMERA == 1

				float fragCoordZ = texture2D( tDepth, screenPosition ).x;
				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );

			#else

				return texture2D( tDepth, screenPosition ).x;

			#endif

		}

		float getViewZ( const in float depth ) {

			#if PERSPECTIVE_CAMERA == 1

				return perspectiveDepthToViewZ( depth, cameraNear, cameraFar );

			#else

				return orthographicDepthToViewZ( depth, cameraNear, cameraFar );

			#endif

		}

		vec3 getViewPosition( const in vec2 screenPosition, const in float depth, const in float viewZ ) {

			float clipW = cameraProjectionMatrix[2][3] * viewZ + cameraProjectionMatrix[3][3];

			vec4 clipPosition = vec4( ( vec3( screenPosition, depth ) - 0.5 ) * 2.0, 1.0 );

			clipPosition *= clipW;

			return ( cameraInverseProjectionMatrix * clipPosition ).xyz;

		}

		vec3 getViewNormal( const in vec2 screenPosition ) {

			return unpackRGBToNormal( texture2D( tNormal, screenPosition ).xyz );

		}

		void main() {

			float depth = getDepth( vUv );

			if ( depth == 1.0 ) {

				gl_FragColor = vec4( 1.0 );

			} else {

				float viewZ = getViewZ( depth );

				vec3 viewPosition = getViewPosition( vUv, depth, viewZ );
				vec3 viewNormal = getViewNormal( vUv );

				vec2 noiseScale = vec2( resolution.x / 4.0, resolution.y / 4.0 );
				vec3 random = vec3( texture2D( tNoise, vUv * noiseScale ).r );



				vec3 tangent = normalize( random - viewNormal * dot( random, viewNormal ) );
				vec3 bitangent = cross( viewNormal, tangent );
				mat3 kernelMatrix = mat3( tangent, bitangent, viewNormal );

				float occlusion = 0.0;

				for ( int i = 0; i < KERNEL_SIZE; i ++ ) {

					vec3 sampleVector = kernelMatrix * kernel[ i ];
					vec3 samplePoint = viewPosition + ( sampleVector * kernelRadius );

					vec4 samplePointNDC = cameraProjectionMatrix * vec4( samplePoint, 1.0 );
					samplePointNDC /= samplePointNDC.w;

					vec2 samplePointUv = samplePointNDC.xy * 0.5 + 0.5;

					float realDepth = getLinearDepth( samplePointUv );
					float sampleDepth = viewZToOrthographicDepth( samplePoint.z, cameraNear, cameraFar );
					float delta = sampleDepth - realDepth;

					if ( delta > minDistance && delta < maxDistance ) {

						occlusion += 1.0;

					}

				}

				occlusion = clamp( occlusion / float( KERNEL_SIZE ), 0.0, 1.0 );

				gl_FragColor = vec4( vec3( 1.0 - occlusion ), 1.0 );

			}

		}`},ie={name:"SSAODepthShader",defines:{PERSPECTIVE_CAMERA:1},uniforms:{tDepth:{value:null},cameraNear:{value:null},cameraFar:{value:null}},vertexShader:`varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`uniform sampler2D tDepth;

		uniform float cameraNear;
		uniform float cameraFar;

		varying vec2 vUv;

		#include <packing>

		float getLinearDepth( const in vec2 screenPosition ) {

			#if PERSPECTIVE_CAMERA == 1

				float fragCoordZ = texture2D( tDepth, screenPosition ).x;
				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );

			#else

				return texture2D( tDepth, screenPosition ).x;

			#endif

		}

		void main() {

			float depth = getLinearDepth( vUv );
			gl_FragColor = vec4( vec3( 1.0 - depth ), 1.0 );

		}`},ae={name:"SSAOBlurShader",uniforms:{tDiffuse:{value:null},resolution:{value:new Ee}},vertexShader:`varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`uniform sampler2D tDiffuse;

		uniform vec2 resolution;

		varying vec2 vUv;

		void main() {

			vec2 texelSize = ( 1.0 / resolution );
			float result = 0.0;

			for ( int i = - 2; i <= 2; i ++ ) {

				for ( int j = - 2; j <= 2; j ++ ) {

					vec2 offset = ( vec2( float( i ), float( j ) ) ) * texelSize;
					result += texture2D( tDiffuse, vUv + offset ).r;

				}

			}

			gl_FragColor = vec4( vec3( result / ( 5.0 * 5.0 ) ), 1.0 );

		}`};var Ae=class V extends Xe{constructor(e,t,i=512,a=512,r=32){super(),this.width=i,this.height=a,this.clear=!0,this.needsSwap=!1,this.camera=t,this.scene=e,this.kernelRadius=8,this.kernel=[],this.noiseTexture=null,this.output=0,this.minDistance=.005,this.maxDistance=.1,this._visibilityCache=[],this._generateSampleKernel(r),this._generateRandomKernelRotations();let o=new Ge;o.format=ze,o.type=Ie,this.normalRenderTarget=new je(this.width,this.height,{minFilter:be,magFilter:be,type:Ne,depthTexture:o}),this.ssaoRenderTarget=new je(this.width,this.height,{type:Ne}),this.blurRenderTarget=this.ssaoRenderTarget.clone(),this.ssaoMaterial=new ee({defines:Object.assign({},te.defines),uniforms:$.clone(te.uniforms),vertexShader:te.vertexShader,fragmentShader:te.fragmentShader,blending:Z}),this.ssaoMaterial.defines.KERNEL_SIZE=r,this.ssaoMaterial.uniforms.tNormal.value=this.normalRenderTarget.texture,this.ssaoMaterial.uniforms.tDepth.value=this.normalRenderTarget.depthTexture,this.ssaoMaterial.uniforms.tNoise.value=this.noiseTexture,this.ssaoMaterial.uniforms.kernel.value=this.kernel,this.ssaoMaterial.uniforms.cameraNear.value=this.camera.near,this.ssaoMaterial.uniforms.cameraFar.value=this.camera.far,this.ssaoMaterial.uniforms.resolution.value.set(this.width,this.height),this.ssaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.ssaoMaterial.uniforms.cameraInverseProjectionMatrix.value.copy(this.camera.projectionMatrixInverse),this.normalMaterial=new Qe,this.normalMaterial.blending=Z,this.blurMaterial=new ee({defines:Object.assign({},ae.defines),uniforms:$.clone(ae.uniforms),vertexShader:ae.vertexShader,fragmentShader:ae.fragmentShader}),this.blurMaterial.uniforms.tDiffuse.value=this.ssaoRenderTarget.texture,this.blurMaterial.uniforms.resolution.value.set(this.width,this.height),this.depthRenderMaterial=new ee({defines:Object.assign({},ie.defines),uniforms:$.clone(ie.uniforms),vertexShader:ie.vertexShader,fragmentShader:ie.fragmentShader,blending:Z}),this.depthRenderMaterial.uniforms.tDepth.value=this.normalRenderTarget.depthTexture,this.depthRenderMaterial.uniforms.cameraNear.value=this.camera.near,this.depthRenderMaterial.uniforms.cameraFar.value=this.camera.far,this.copyMaterial=new ee({uniforms:$.clone(le.uniforms),vertexShader:le.vertexShader,fragmentShader:le.fragmentShader,transparent:!0,depthTest:!1,depthWrite:!1,blendSrc:Ve,blendDst:_e,blendEquation:Re,blendSrcAlpha:Ze,blendDstAlpha:_e,blendEquationAlpha:Re}),this._fsQuad=new Ye(null),this._originalClearColor=new We}dispose(){this.normalRenderTarget.dispose(),this.ssaoRenderTarget.dispose(),this.blurRenderTarget.dispose(),this.normalMaterial.dispose(),this.blurMaterial.dispose(),this.copyMaterial.dispose(),this.depthRenderMaterial.dispose(),this._fsQuad.dispose()}render(e,t,i){switch(this._overrideVisibility(),this._renderOverride(e,this.normalMaterial,this.normalRenderTarget,7829503,1),this._restoreVisibility(),this.ssaoMaterial.uniforms.kernelRadius.value=this.kernelRadius,this.ssaoMaterial.uniforms.minDistance.value=this.minDistance,this.ssaoMaterial.uniforms.maxDistance.value=this.maxDistance,this._renderPass(e,this.ssaoMaterial,this.ssaoRenderTarget),this._renderPass(e,this.blurMaterial,this.blurRenderTarget),this.output){case V.OUTPUT.SSAO:this.copyMaterial.uniforms.tDiffuse.value=this.ssaoRenderTarget.texture,this.copyMaterial.blending=Z,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:i);break;case V.OUTPUT.Blur:this.copyMaterial.uniforms.tDiffuse.value=this.blurRenderTarget.texture,this.copyMaterial.blending=Z,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:i);break;case V.OUTPUT.Depth:this._renderPass(e,this.depthRenderMaterial,this.renderToScreen?null:i);break;case V.OUTPUT.Normal:this.copyMaterial.uniforms.tDiffuse.value=this.normalRenderTarget.texture,this.copyMaterial.blending=Z,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:i);break;case V.OUTPUT.Default:this.copyMaterial.uniforms.tDiffuse.value=this.blurRenderTarget.texture,this.copyMaterial.blending=Fe,this._renderPass(e,this.copyMaterial,this.renderToScreen?null:i);break;default:console.warn("THREE.SSAOPass: Unknown output type.")}}setSize(e,t){this.width=e,this.height=t,this.ssaoRenderTarget.setSize(e,t),this.normalRenderTarget.setSize(e,t),this.blurRenderTarget.setSize(e,t),this.ssaoMaterial.uniforms.resolution.value.set(e,t),this.ssaoMaterial.uniforms.cameraProjectionMatrix.value.copy(this.camera.projectionMatrix),this.ssaoMaterial.uniforms.cameraInverseProjectionMatrix.value.copy(this.camera.projectionMatrixInverse),this.blurMaterial.uniforms.resolution.value.set(e,t)}_renderPass(e,t,i,a,r){e.getClearColor(this._originalClearColor);let o=e.getClearAlpha(),s=e.autoClear;e.setRenderTarget(i),e.autoClear=!1,a!=null&&(e.setClearColor(a),e.setClearAlpha(r||0),e.clear()),this._fsQuad.material=t,this._fsQuad.render(e),e.autoClear=s,e.setClearColor(this._originalClearColor),e.setClearAlpha(o)}_renderOverride(e,t,i,a,r){e.getClearColor(this._originalClearColor);let o=e.getClearAlpha(),s=e.autoClear;e.setRenderTarget(i),e.autoClear=!1,a=t.clearColor||a,r=t.clearAlpha||r,a!=null&&(e.setClearColor(a),e.setClearAlpha(r||0),e.clear()),this.scene.overrideMaterial=t,e.render(this.scene,this.camera),this.scene.overrideMaterial=null,e.autoClear=s,e.setClearColor(this._originalClearColor),e.setClearAlpha(o)}_generateSampleKernel(e){let t=this.kernel;for(let i=0;i<e;i++){let a=new qe;a.x=Math.random()*2-1,a.y=Math.random()*2-1,a.z=Math.random(),a.normalize();let r=i/e;r=Ke.lerp(.1,1,r*r),a.multiplyScalar(r),t.push(a)}}_generateRandomKernelRotations(){let i=new ce,a=16,r=new Float32Array(a);for(let o=0;o<a;o++){let s=Math.random()*2-1,b=Math.random()*2-1,n=0;r[o]=i.noise3d(s,b,n)}this.noiseTexture=new Be(r,4,4,Le,Oe),this.noiseTexture.wrapS=ye,this.noiseTexture.wrapT=ye,this.noiseTexture.needsUpdate=!0}_overrideVisibility(){let e=this.scene,t=this._visibilityCache;e.traverse(function(i){(i.isPoints||i.isLine||i.isLine2)&&i.visible&&(i.visible=!1,t.push(i))})}_restoreVisibility(){let e=this._visibilityCache;for(let t=0;t<e.length;t++)e[t].visible=!0;e.length=0}};Ae.OUTPUT={Default:0,SSAO:1,Blur:2,Depth:3,Normal:4};export{Ae as SSAOPass};
