Shader "Custom/Highlight"
{
	Properties
	{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}
		_OutlineColor("Outline Color", Color) = (1,1,1,1)
		_OutlineWidth("Outline Width", Float) = 0.1
	}

		SubShader
		{
			Pass
			{
				Tags { "RenderType" = "Transparent" "Queue" = "Transparent" "IgnoreProjector" = "True" }

				Blend SrcAlpha OneMinusSrcAlpha
				Cull Front


				CGPROGRAM
				#pragma vertex vert
				#pragma fragment frag
				#pragma fragmentoption ARB_precision_hint_fastest
				#include "UnityCG.cginc"

				struct appdata
				{
					float4 vertex : POSITION;
					float3 normal : NORMAL;
				};

				struct v2f
				{
					float4 pos : SV_POSITION;
				};

				float _OutlineWidth;

				v2f vert(appdata v)
				{
					v2f o;
					float3 vertex = v.vertex;
					vertex.xy *= (1 + _OutlineWidth);
					o.pos = UnityObjectToClipPos(vertex);
					return o;
				}

				half4 _OutlineColor;

				half4 frag(v2f i) : COLOR
				{
					return _OutlineColor;
				}

				ENDCG
			}

			Pass
			{
				Tags { "RenderType" = "Opaque" "Queue" = "Opaque" "IgnoreProjector" = "True" }
				LOD 200

				CGPROGRAM
				#pragma vertex vert
				#pragma fragment frag
				#pragma fragmentoption ARB_precision_hint_fastest
				#include "UnityCG.cginc"

				struct appdata
				{
					float4 vertex : POSITION;
					float2 uv : TEXCOORD0;
				};

				struct v2f
				{
					float4 pos : SV_POSITION;
					float2 uv : TEXCOORD0;
				};

				v2f vert(appdata v)
				{
					v2f o;
					o.pos = UnityObjectToClipPos(v.vertex);
					o.uv = v.uv;
					return o;
				}

				half4 _Color;
				sampler2D _MainTex;

				half4 frag(v2f i) : COLOR
				{
					return _Color * tex2D(_MainTex, i.uv);
				}

				ENDCG
			}
		}

			FallBack "Diffuse"
}
