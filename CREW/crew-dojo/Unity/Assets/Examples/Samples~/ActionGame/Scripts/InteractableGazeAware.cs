//-----------------------------------------------------------------------
// Copyright 2016 Tobii AB (publ). All rights reserved.
//-----------------------------------------------------------------------

using System.Collections;
using UnityEngine;

namespace Tobii.Gaming.Examples.ActionGame
{
	public class InteractableGazeAware : GazeAware
	{
		public string Label;
		private bool _running;

		public virtual void Interact()
		{
			if (!_running)
				StartCoroutine(Pickup());
		}

		private IEnumerator Pickup()
		{
			var oldScale = transform.localScale;
			_running = true;
			while (transform.localScale.magnitude > 0.001f)
			{
				transform.localScale = Vector3.Lerp(transform.localScale, Vector3.zero, Time.deltaTime * 10f);
				yield return null;
			}

			yield return new WaitForSeconds(2);
			transform.localScale = oldScale;
			_running = false;
		}
	}
}
