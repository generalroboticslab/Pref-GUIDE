//-----------------------------------------------------------------------
// Copyright 2016 Tobii AB (publ). All rights reserved.
//-----------------------------------------------------------------------

using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

namespace Tobii.Gaming.Examples.ActionGame
{
/*
* This is the specialization for Extended View when in third person.
* This version of Extended View orbits around a point. This point is usually somewhere close to the player character.
*/
	public class ExtendedViewThirdPerson : ExtendedView
	{
		public const int RaycastLayerMask = ~0x24; //0b100100;           // ignore "ignore raycast" and "ui" layers

		public Transform OrbitPoint;
		public SimpleMoveController YawController;
		public SimpleMoveController PitchController;
		public WeaponController WeaponController;

		[Tooltip("Reference to a crosshair Image. The supplied crosshair will be moved according to the character forward direction rather than stay at the center of screen.")]
		public Image Crosshair;

		public float MinimumPitch = -90;
		public float MaximumPitch = 90;
		public float ZoomDistance = 2;

		private Vector3 _crosshairScreenPosition;
		private Transform _worldCenterCrosshairTransformProjected;
		private Camera _usedCamera;
		private Quaternion _localRotation = Quaternion.identity;
		
		public Transform WorldCenterCrosshairTransformProjected
		{
			get
			{
				if (_worldCenterCrosshairTransformProjected != null)
					return _worldCenterCrosshairTransformProjected.transform;

				_worldCenterCrosshairTransformProjected = new GameObject("WorldCenterCrosshairProjected").transform;
				return _worldCenterCrosshairTransformProjected.transform;
			}
		}

		protected override void Start()
		{
			base.Start();
			RenderPipelineManager.beginFrameRendering += RenderPipelineManagerOnbeginFrameRendering;
			RenderPipelineManager.endCameraRendering += RenderPipelineManagerOnendCameraRendering;
			
			_usedCamera = GetComponent<Camera>();
		}

		protected override void UpdateTransform()
		{
			if (WeaponController != null)
			{
				IsAiming = WeaponController.IsAiming;
			}
		}

		void OnDestroy()
		{
			RenderPipelineManager.beginFrameRendering -= RenderPipelineManagerOnbeginFrameRendering;
			RenderPipelineManager.endCameraRendering -= RenderPipelineManagerOnendCameraRendering;
		}

		void OnPreCull()
		{
			RenderPipelineManagerOnbeginFrameRendering(new ScriptableRenderContext(), new[] {_usedCamera});
			StartCoroutine(ResetCameraLocal(_localRotation, OrbitPoint));
		}
		
		private void RenderPipelineManagerOnbeginFrameRendering(ScriptableRenderContext arg1, Camera[] arg2)
		{
			_localRotation = OrbitPoint.localRotation;

			transform.position = OrbitPoint.position - OrbitPoint.forward * ZoomDistance;
			transform.rotation = OrbitPoint.rotation;

			UpdateCameraWithoutExtendedView(_usedCamera);
			var worldUp = Vector3.up;
			Rotate(OrbitPoint, up: worldUp);

			transform.position = OrbitPoint.position - OrbitPoint.forward * ZoomDistance;
			transform.rotation = OrbitPoint.rotation;

			UpdateCameraWithExtendedView(_usedCamera);

			UpdateCrosshair();
		}
		
		private void RenderPipelineManagerOnendCameraRendering(ScriptableRenderContext arg1, Camera arg2)
		{
			OrbitPoint.localRotation = _localRotation;
		}

		public override void AimAtWorldPosition(Vector3 worldPostion)
		{
			if (WeaponController != null)
			{
				IsAiming = WeaponController.IsAiming;
			}

			//Direction betweeon aim position and orbit point
			var direction = worldPostion - OrbitPoint.position;
			var rotation = Quaternion.LookRotation(direction);

			InitAimAtGazeOffset(Mathf.DeltaAngle(rotation.eulerAngles.y, _usedCamera.transform.rotation.eulerAngles.y),
				Mathf.DeltaAngle(rotation.eulerAngles.x, _usedCamera.transform.rotation.eulerAngles.x));

			if (YawController != null
			    && PitchController != null)
			{
				YawController.SetRotation(rotation);
				PitchController.SetRotation(rotation);
			}
		}

		private void UpdateCrosshair()
		{
			if (Crosshair == null) return;

			RaycastHit hitInfo;
			if (Physics.Raycast(CameraWithoutExtendedView.transform.position,
				CameraWithoutExtendedView.transform.forward, out hitInfo, 40, RaycastLayerMask))
			{
				WorldCenterCrosshairTransformProjected.position = hitInfo.point;
			}
			else
			{
				WorldCenterCrosshairTransformProjected.position = CameraWithoutExtendedView.transform.position +
				                                                  CameraWithoutExtendedView.transform.forward * 1000;
			}

			if (IsAiming)
			{
				_crosshairScreenPosition =
					CameraWithExtendedView.WorldToScreenPoint(WorldCenterCrosshairTransformProjected.position);
			}
			else
			{
				_crosshairScreenPosition = Vector3.Lerp(_crosshairScreenPosition,
					CameraWithExtendedView.WorldToScreenPoint(WorldCenterCrosshairTransformProjected.position),
					Time.unscaledDeltaTime * 50);
			}

			var canvas = Crosshair.GetComponentInParent<Canvas>();

			Crosshair.rectTransform.anchoredPosition =
				new Vector2(
					(_crosshairScreenPosition.x - Screen.width * 0.5f) *
					(canvas.GetComponent<RectTransform>().sizeDelta.x / Screen.width),
					(_crosshairScreenPosition.y - Screen.height * 0.5f) *
					(canvas.GetComponent<RectTransform>().sizeDelta.y / Screen.height));
		}
	}
}
