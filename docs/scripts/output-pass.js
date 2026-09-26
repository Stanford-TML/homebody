import{a as u,b as h}from"./pass.js";import{$ as t,Cd as f,Uc as g,_ as a,_b as p,_e as _,aa as o,ba as n,ca as r,da as l,ea as s}from"./three-core.js";var i={name:"OutputShader",uniforms:{tDiffuse:{value:null},toneMappingExposure:{value:1}},vertexShader:`
		precision highp float;

		uniform mat4 modelViewMatrix;
		uniform mat4 projectionMatrix;

		attribute vec3 position;
		attribute vec2 uv;

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`

		precision highp float;

		uniform sampler2D tDiffuse;

		#include <tonemapping_pars_fragment>
		#include <colorspace_pars_fragment>

		varying vec2 vUv;

		void main() {

			gl_FragColor = texture2D( tDiffuse, vUv );



			#ifdef LINEAR_TONE_MAPPING

				gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );

			#elif defined( REINHARD_TONE_MAPPING )

				gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );

			#elif defined( CINEON_TONE_MAPPING )

				gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );

			#elif defined( ACES_FILMIC_TONE_MAPPING )

				gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );

			#elif defined( AGX_TONE_MAPPING )

				gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );

			#elif defined( NEUTRAL_TONE_MAPPING )

				gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );

			#elif defined( CUSTOM_TONE_MAPPING )

				gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );

			#endif



			#ifdef SRGB_TRANSFER

				gl_FragColor = sRGBTransferOETF( gl_FragColor );

			#endif

		}`};var M=class extends u{constructor(){super(),this.uniforms=f.clone(i.uniforms),this.material=new _({name:i.name,uniforms:this.uniforms,vertexShader:i.vertexShader,fragmentShader:i.fragmentShader}),this._fsQuad=new h(this.material),this._outputColorSpace=null,this._toneMapping=null}render(e,m,d){this.uniforms.tDiffuse.value=d.texture,this.uniforms.toneMappingExposure.value=e.toneMappingExposure,(this._outputColorSpace!==e.outputColorSpace||this._toneMapping!==e.toneMapping)&&(this._outputColorSpace=e.outputColorSpace,this._toneMapping=e.toneMapping,this.material.defines={},g.getTransfer(this._outputColorSpace)===p&&(this.material.defines.SRGB_TRANSFER=""),this._toneMapping===a?this.material.defines.LINEAR_TONE_MAPPING="":this._toneMapping===t?this.material.defines.REINHARD_TONE_MAPPING="":this._toneMapping===o?this.material.defines.CINEON_TONE_MAPPING="":this._toneMapping===n?this.material.defines.ACES_FILMIC_TONE_MAPPING="":this._toneMapping===l?this.material.defines.AGX_TONE_MAPPING="":this._toneMapping===s?this.material.defines.NEUTRAL_TONE_MAPPING="":this._toneMapping===r&&(this.material.defines.CUSTOM_TONE_MAPPING=""),this.material.needsUpdate=!0),this.renderToScreen===!0?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(m),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this._fsQuad.render(e))}dispose(){this.material.dispose(),this._fsQuad.dispose()}};export{M as OutputPass};
