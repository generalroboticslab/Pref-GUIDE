//-----------------------------------------------------------------------
// Copyright 2016 Tobii AB (publ). All rights reserved.
//-----------------------------------------------------------------------

using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

namespace Tobii.Gaming.Examples.ActionGame
{
	/// <summary>
	/// This is the specialization for Extended View when in first person.
	/// This should be a direct child of the transform that is responsible for mouse yaw and pitch shifts, or it won't work.
	/// The camera used for the first person controller should also be a child of the transform associated with this component.
	/// </summary>
	public class ExtendedViewFirstPerson : ExtendedView
	{
		[Tooltip("Reference to a crosshair Image. The supplied crosshair will be moved according to the character forward direction rather than stay at the center of screen.")]
		public Image Crosshair;

		private Camera _usedCamera;

		private Vector3 _crosshairScreenPosition;
		private WeaponController _weaponController;
		private SimpleMoveController SimpleMoveController { get; set; }

		private Quaternion _localRotation = Quaternion.identity;
		
		private void Awake()
		{
			SimpleMoveController = GetComponentInParent<SimpleMoveController>();
		}

		protected override void Start()
		{
			base.Start();
			
			RenderPipelineManager.beginFrameRendering += RenderPipelineManagerOnbeginFrameRendering;
			RenderPipelineManager.endCameraRendering += RenderPipelineManagerOnendCameraRendering;
			
			// Make sure that our transform is in the identity state when we start.
			transform.localRotation = Quaternion.identity;
			transform.localPosition = Vector3.zero;

			_usedCamera = GetComponent<Camera>();
			_weaponController = GetComponentInParent<WeaponController>();
		}

		protected override void UpdateTransform()
		{
			if (_weaponController != null)
			{
				IsAiming = _weaponController.IsAiming;
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
			StartCoroutine(ResetCameraLocal(_localRotation, transform));
		}
		
		private void RenderPipelineManagerOnbeginFrameRendering(ScriptableRenderContext arg1, Camera[] arg2)
		{
			var crosshairWorldPosition = _usedCamera.transform.position + _usedCamera.transform.forward;
			_localRotation = transform.localRotation;
			UpdateCameraWithoutExtendedView(_usedCamera);
			Rotate(transform);
			UpdateCameraWithExtendedView(_usedCamera);

			_crosshairScreenPosition = _usedCamera.WorldToScreenPoint(crosshairWorldPosition);
			UpdateCrosshair();
		}
		
		private void RenderPipelineManagerOnendCameraRendering(ScriptableRenderContext arg1, Camera arg2)
		{
			transform.localRotation = _localRotation;
		}
		
		private void UpdateCrosshair()
		{
			if (Crosshair == null) return;

			var canvas = Crosshair.GetComponentInParent<Canvas>();

			Crosshair.rectTransform.anchoredPosition =
				new Vector2(
					(_crosshairScreenPosition.x - Screen.width * 0.5f) *
					(canvas.GetComponent<RectTransform>().sizeDelta.x / Screen.width),
					(_crosshairScreenPosition.y - Screen.height * 0.5f) *
					(canvas.GetComponent<RectTransform>().sizeDelta.y / Screen.height));
		}

		/// <summary>
		/// Aim at Gaze: ...
		/// If you are calling this function manually, make sure that IsAiming property is updated on Update()
		/// </summary>
		/// <param name="worldPostion"></param>
		public override void AimAtWorldPosition(Vector3 worldPostion)
		{
			if (_weaponController != null)
			{
				IsAiming = _weaponController.IsAiming;
			}

			if (SimpleMoveController == null)
			{
				return;
			}

			var direction = worldPostion - transform.position;
			var desiredRotation = Quaternion.LookRotation(direction);

			InitAimAtGazeOffset(
				Mathf.DeltaAngle(desiredRotation.eulerAngles.y,
					CameraWithExtendedView.transform.rotation.eulerAngles.y),
				Mathf.DeltaAngle(desiredRotation.eulerAngles.x,
					CameraWithExtendedView.transform.rotation.eulerAngles.x));

			SimpleMoveController.SetRotation(desiredRotation);
		}
	}
}
