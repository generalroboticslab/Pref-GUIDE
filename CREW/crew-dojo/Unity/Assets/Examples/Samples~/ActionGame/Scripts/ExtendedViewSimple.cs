//-----------------------------------------------------------------------
// Copyright 2016 Tobii AB (publ). All rights reserved.
//-----------------------------------------------------------------------

using UnityEngine;

namespace Tobii.Gaming.Examples.ActionGame
{
/*
* This is the specialization for Extended View when in first person.
*/
	public class ExtendedViewSimple : ExtendedView
	{
		private Camera _usedCamera;

		protected override void Start()
		{
			base.Start();
			_usedCamera = GetComponentInChildren<Camera>();
		}

		protected override void UpdateTransform()
		{
			var localRotation = transform.localRotation;

			UpdateCameraWithoutExtendedView(_usedCamera);
			var worldUp = Vector3.up;
			Rotate(transform, up: worldUp);
			UpdateCameraWithExtendedView(_usedCamera);

			StartCoroutine(ResetCameraLocal(localRotation, transform));
		}
	}
}
